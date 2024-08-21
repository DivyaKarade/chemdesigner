import streamlit as st
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import io  # Imported io module to capture model summaries
from io import BytesIO

# Page expands to full width
st.set_page_config(page_title='AIDrugApp', page_icon='🌐', layout="wide")

st.title('ChemDesigner: Molecule Graph Generator')
st.success(
    "This module of [**AIDrugApp v1.2.6**](https://aidrugapp.streamlit.app/) uses Generative Adversarial Network with Gradient Penalty algorithms (WGAN-GP) and Graph Neural Network (GNN) for generating new chemical structures.")

expander_bar = st.expander("👉 More information")
expander_bar.markdown("""
        * **Python libraries:** Tensorflow, Keras, RDKit, streamlit, pandas, numpy
        """)

expander_bar = st.expander("👉 How to use ChemDesigner?")
expander_bar.markdown("""**Step 1:** In the User input-side panel, upload your input CSV file containing SMILES data. (*Example input file provided*)""")
expander_bar.markdown("""**Step 2:** In the "Select Atoms" section, choose the atoms you want to include in your analysis. (*Ensure that your SMILES strings in the uploaded CSV file contain only the atoms you selected in the "Select Atoms" section.*)""")
expander_bar.markdown("""**Step 3:** In the "Select Bonds" section, choose the types of chemical bonds to include.""")
expander_bar.markdown("""**Step 4:** In the "Set Parameters" section, adjust the hyperparameters for training the GAN.""")
expander_bar.markdown("""**Step 5:** Click the "✨ Generate Molecules" button in the sidebar to start the molecule generation process.""")

"""---"""

st.sidebar.header('⚙️ USER INPUT PANEL')

# Disable RDKit logging
RDLogger.DisableLog("rdApp.*")


# Sidebar for CSV upload
with st.sidebar:
    st.subheader('1. Upload your CSV data')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file containing SMILES", type=["csv"])
    st.sidebar.markdown("""
        [Example CSV input file](https://github.com/DivyaKarade/Example-input-file-for-ChemDesigner/blob/main/Example%20input%20file%20for%20ChemDesigner.csv)
    """)

# Define default atom mapping with atom names only
default_atom_mapping = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
    "Cl": 4,
    "S": 5,
    "Br": 6,
    "Zn": 7,
    "I": 8,
    "P": 9,
    "B": 10
}

# Atom selection
with st.sidebar:
    st.subheader("2. Select Atoms")
    selected_atoms = st.multiselect(
        "Choose atoms to include:",
        options=list(default_atom_mapping.keys()),
        default=list(default_atom_mapping.keys())  # Default to all atoms
    )

# Update atom mapping based on user selection
atom_mapping = {atom: idx for idx, atom in enumerate(selected_atoms)}
atom_mapping_rev = {idx: atom for atom, idx in atom_mapping.items()}

# Define default bond mapping
default_bond_mapping = {
    "SINGLE": Chem.BondType.SINGLE,
    "DOUBLE": Chem.BondType.DOUBLE,
    "TRIPLE": Chem.BondType.TRIPLE,
    "AROMATIC": Chem.BondType.AROMATIC,
}

# Bond selection
with st.sidebar:
    st.subheader("3. Select Bonds")
    selected_bonds = st.multiselect(
        "Choose bond types to include:",
        options=list(default_bond_mapping.keys()),
        default=list(default_bond_mapping.keys())  # Default to all bond types
    )

# Update bond mapping based on user selection
bond_mapping = {bond: idx for idx, bond in enumerate(selected_bonds)}
bond_mapping_rev = {idx: bond for bond, idx in bond_mapping.items()}


# Function to load CSV data
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


# Initialize data as an empty DataFrame
data = pd.DataFrame()

if uploaded_file is not None:
    df4 = load_csv(uploaded_file)
    data = df4.iloc[:, 0:]
    X = data
    # Write CSV data
    data.to_csv('molecule.smi', sep='\t', header=False, index=False)
    st.subheader('Uploaded data')
    st.write(data)
    smiles_list = data['smiles'].values
else:
    st.info('Awaiting .csv file to be uploaded')
    smiles_list = []  # Initialize as an empty list if no file is uploaded

# Let's look at a molecule of the dataset
if len(smiles_list) > 0:  # Explicitly check if the list is not empty
    smiles = smiles_list[0]  # Select the first SMILES string from the list
    molecule = Chem.MolFromSmiles(smiles)

# Convert SMILES to RDKit Mol objects
molecules = [Chem.MolFromSmiles(sm) for sm in smiles_list if Chem.MolFromSmiles(sm)]

# Ensure all atoms in the SMILES strings are in the atom mapping
for mol in molecules:
    if mol:
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in atom_mapping:
                raise ValueError(f"Atom symbol '{atom_symbol}' not in atom_mapping.")

# Calculate NUM_ATOMS based on the loaded molecules
NUM_ATOMS = max([mol.GetNumAtoms() for mol in molecules]) if molecules else 0
ATOM_DIM = len(atom_mapping)  # Number of atom types, including only selected atoms
BOND_DIM = len(bond_mapping)  # Number of bond types
LATENT_DIM = 64  # Size of the latent space

# Sidebar to select hyperparameters
st.sidebar.subheader("4. Set Parameters")
epochs = st.sidebar.slider("Select number of training epochs:", min_value=1, max_value=100, value=10, step=1)
batch_size = st.sidebar.slider("Select batch size:", min_value=1, max_value=64, value=16, step=1)
num_samples = st.sidebar.slider("Select number of molecules to display:", min_value=1, max_value=50, value=25, step=1)


def smiles_to_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string.")

    adjacency = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
    features = np.zeros((NUM_ATOMS, ATOM_DIM), "float32")

    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        if atom_symbol not in atom_mapping:
            raise ValueError(f"Atom symbol '{atom_symbol}' not in atom_mapping.")
        atom_type = atom_mapping[atom_symbol]
        features[i] = np.eye(ATOM_DIM)[atom_type]
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j)
            bond_type_name = bond.GetBondType().name
            if bond_type_name not in bond_mapping:
                raise ValueError(f"Bond type '{bond_type_name}' not in bond_mapping.")
            bond_type_idx = bond_mapping[bond_type_name]
            if bond_type_idx >= BOND_DIM:
                raise IndexError(f"Bond type index '{bond_type_idx}' is out of bounds.")
            adjacency[bond_type_idx, i, j] = 1
            adjacency[bond_type_idx, j, i] = 1

    # Handle isolated atoms and bonds
    adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1

    return adjacency, features


def graph_to_molecule(graph):
    # Unpack graph
    adjacency, features = graph

    # Create a molecule object
    molecule = Chem.RWMol()

    # Remove "no atoms" & atoms with no bonds
    keep_idx = np.where(
        (np.argmax(features, axis=1) != ATOM_DIM - 1)
        & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
    )[0]
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to molecule if they are in the atom mapping
    atom_indices = {}
    for atom_type_idx in np.argmax(features, axis=1):
        atom_symbol = atom_mapping_rev.get(atom_type_idx)
        if atom_symbol:
            atom_idx = molecule.AddAtom(Chem.Atom(atom_symbol))
            atom_indices[atom_idx] = atom_type_idx

    # Add bonds between atoms in the molecule
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == BOND_DIM - 1:
            continue
        bond_type_name = bond_mapping_rev.get(bond_ij)
        if bond_type_name and atom_i in atom_indices and atom_j in atom_indices:
            bond_type = default_bond_mapping.get(bond_type_name)  # Convert bond type name to BondType enum
            molecule.AddBond(int(atom_i), int(atom_j), bond_type)

    # Sanitize the molecule; return None if sanitization fails
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return molecule


# Button to sample new molecules
if st.sidebar.button("✨ Generate Molecules"):

    # Test helper functions
    if not data.empty:
        graph_to_molecule(smiles_to_graph(smiles))

        # To save training time, we'll only use a tenth of the dataset.
        adjacency_tensor, feature_tensor = [], []
        for smiles in data['smiles'][::10]:
            adjacency, features = smiles_to_graph(smiles)
            adjacency_tensor.append(adjacency)
            feature_tensor.append(features)

        adjacency_tensor = np.array(adjacency_tensor)
        feature_tensor = np.array(feature_tensor)

        st.subheader("Tensor Shapes")
        st.write(f"**Adjacency Tensor Shape:** {adjacency_tensor.shape}")
        st.write(f"**Feature Tensor Shape:** {feature_tensor.shape}")

    # Precompute the sizes for Dense layers
    adjacency_size = np.prod((BOND_DIM, NUM_ATOMS, NUM_ATOMS))
    feature_size = np.prod((NUM_ATOMS, ATOM_DIM))


    class TransposeAndAverageLayer(keras.layers.Layer):
        def __init__(self, **kwargs):
            super(TransposeAndAverageLayer, self).__init__(**kwargs)

        def call(self, x_adjacency):
            x_adjacency_transposed = tf.transpose(x_adjacency, perm=[0, 1, 3, 2])
            return (x_adjacency + x_adjacency_transposed) / 2


    def GraphGenerator(
            dense_units,
            dropout_rate,
            latent_dim,
            adjacency_shape,
            feature_shape,
    ):
        z = keras.layers.Input(shape=(latent_dim,))
        x = z
        for units in dense_units:
            x = keras.layers.Dense(units, activation="tanh")(x)
            x = keras.layers.Dropout(dropout_rate)(x)

        x_adjacency = keras.layers.Dense(adjacency_size)(x)
        x_adjacency = keras.layers.Reshape(adjacency_shape)(x_adjacency)
        x_adjacency = TransposeAndAverageLayer()(x_adjacency)
        x_adjacency = keras.layers.Softmax(axis=1)(x_adjacency)

        x_features = keras.layers.Dense(feature_size)(x)
        x_features = keras.layers.Reshape(feature_shape)(x_features)
        x_features = keras.layers.Softmax(axis=2)(x_features)

        return keras.Model(inputs=z, outputs=[x_adjacency, x_features], name="Generator")


    generator = GraphGenerator(
        dense_units=[128, 256, 512],
        dropout_rate=0.2,
        latent_dim=LATENT_DIM,
        adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
        feature_shape=(NUM_ATOMS, ATOM_DIM),
    )
    generator.summary()

    # Capture and display Generator summary
    st.subheader("Generator Model Summary")
    generator_summary = io.StringIO()
    generator.summary(print_fn=lambda x: generator_summary.write(x + "\n"))
    st.code(generator_summary.getvalue(), language='python')


    class RelationalGraphConvLayer(keras.layers.Layer):
        def __init__(
                self,
                units=128,
                activation="relu",
                use_bias=False,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                **kwargs
        ):
            super().__init__(**kwargs)

            self.units = units
            self.activation = keras.activations.get(activation)
            self.use_bias = use_bias
            self.kernel_initializer = keras.initializers.get(kernel_initializer)
            self.bias_initializer = keras.initializers.get(bias_initializer)
            self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
            self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        def build(self, input_shape):
            bond_dim = input_shape[0][1]
            atom_dim = input_shape[1][2]

            self.kernel = self.add_weight(
                shape=(bond_dim, atom_dim, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
                name="W",
                dtype=tf.float32,
            )

            if self.use_bias:
                self.bias = self.add_weight(
                    shape=(bond_dim, 1, self.units),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=True,
                    name="b",
                    dtype=tf.float32,
                )

            self.built = True

        def call(self, inputs, training=False):
            adjacency, features = inputs
            # Aggregate information from neighbors
            x = tf.matmul(adjacency, features[:, None, :, :])
            # Apply linear transformation
            x = tf.matmul(x, self.kernel)
            if self.use_bias:
                x += self.bias
            # Reduce bond types dim
            x_reduced = tf.reduce_sum(x, axis=1)
            # Apply non-linear transformation
            return self.activation(x_reduced)


    def GraphDiscriminator(
            gconv_units, dense_units, dropout_rate, adjacency_shape, feature_shape
    ):
        adjacency = keras.layers.Input(shape=adjacency_shape)
        features = keras.layers.Input(shape=feature_shape)

        # Propagate through one or more graph convolutional layers
        features_transformed = features
        for units in gconv_units:
            features_transformed = RelationalGraphConvLayer(units)([adjacency, features_transformed])

        # Reduce 2-D representation of molecule to 1-D
        x = keras.layers.GlobalAveragePooling1D()(features_transformed)

        # Propagate through one or more densely connected layers
        for units in dense_units:
            x = keras.layers.Dense(units, activation="relu")(x)
            x = keras.layers.Dropout(dropout_rate)(x)

        # For each molecule, output a single scalar value expressing the
        # "realness" of the inputted molecule
        x_out = keras.layers.Dense(1, dtype="float32")(x)

        return keras.Model(inputs=[adjacency, features], outputs=x_out)


    discriminator = GraphDiscriminator(
        gconv_units=[128, 128, 128, 128],
        dense_units=[512, 512],
        dropout_rate=0.2,
        adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
        feature_shape=(NUM_ATOMS, ATOM_DIM),
    )
    discriminator.summary()

    # Capture and display Discriminator summary
    st.subheader("Discriminator Model Summary")
    discriminator_summary = io.StringIO()
    discriminator.summary(print_fn=lambda x: discriminator_summary.write(x + "\n"))
    st.code(discriminator_summary.getvalue(), language='python')


    class GraphWGAN(keras.Model):
        def __init__(
                self,
                generator,
                discriminator,
                discriminator_steps=1,
                generator_steps=1,
                gp_weight=10,
                **kwargs
        ):
            super().__init__(**kwargs)
            self.generator = generator
            self.discriminator = discriminator
            self.discriminator_steps = discriminator_steps
            self.generator_steps = generator_steps
            self.gp_weight = gp_weight
            self.latent_dim = self.generator.input_shape[-1]

        def compile(self, optimizer_generator, optimizer_discriminator, **kwargs):
            super().compile(**kwargs)
            self.optimizer_generator = optimizer_generator
            self.optimizer_discriminator = optimizer_discriminator
            self.metric_generator = keras.metrics.Mean(name="loss_gen")
            self.metric_discriminator = keras.metrics.Mean(name="loss_dis")

        def train_step(self, inputs):
            if isinstance(inputs[0], tuple):
                inputs = inputs[0]

            graph_real = inputs

            self.batch_size = tf.shape(inputs[0])[0]

            # Train the discriminator for one or more steps
            for _ in range(self.discriminator_steps):
                z = tf.random.normal((self.batch_size, self.latent_dim))

                with tf.GradientTape() as tape:
                    graph_generated = self.generator(z, training=True)
                    loss = self._loss_discriminator(graph_real, graph_generated)

                grads = tape.gradient(loss, self.discriminator.trainable_weights)
                self.optimizer_discriminator.apply_gradients(
                    zip(grads, self.discriminator.trainable_weights)
                )
                self.metric_discriminator.update_state(loss)

            # Train the generator for one or more steps
            for _ in range(self.generator_steps):
                z = tf.random.normal((self.batch_size, self.latent_dim))

                with tf.GradientTape() as tape:
                    graph_generated = self.generator(z, training=True)
                    loss = self._loss_generator(graph_generated)

                    grads = tape.gradient(loss, self.generator.trainable_weights)
                    self.optimizer_generator.apply_gradients(
                        zip(grads, self.generator.trainable_weights)
                    )
                    self.metric_generator.update_state(loss)

            return {m.name: m.result() for m in self.metrics}

        def _loss_discriminator(self, graph_real, graph_generated):
            logits_real = self.discriminator(graph_real, training=True)
            logits_generated = self.discriminator(graph_generated, training=True)
            loss = tf.reduce_mean(logits_generated) - tf.reduce_mean(logits_real)
            loss_gp = self._gradient_penalty(graph_real, graph_generated)
            return loss + loss_gp * self.gp_weight

        def _loss_generator(self, graph_generated):
            logits_generated = self.discriminator(graph_generated, training=True)
            return -tf.reduce_mean(logits_generated)

        def _gradient_penalty(self, graph_real, graph_generated):
            # Unpack graphs
            adjacency_real, features_real = graph_real
            adjacency_generated, features_generated = graph_generated

            # Generate interpolated graphs (adjacency_interp and features_interp)
            alpha = tf.random.uniform([self.batch_size])
            alpha = tf.reshape(alpha, (self.batch_size, 1, 1, 1))
            adjacency_interp = (adjacency_real * alpha) + (1 - alpha) * adjacency_generated
            alpha = tf.reshape(alpha, (self.batch_size, 1, 1))
            features_interp = (features_real * alpha) + (1 - alpha) * features_generated

            # Compute the logits of interpolated graphs
            with tf.GradientTape() as tape:
                tape.watch(adjacency_interp)
                tape.watch(features_interp)
                logits = self.discriminator(
                    [adjacency_interp, features_interp], training=True
                )

            # Compute the gradients with respect to the interpolated graphs
            grads = tape.gradient(logits, [adjacency_interp, features_interp])
            # Compute the gradient penalty
            grads_adjacency_penalty = (1 - tf.norm(grads[0], axis=1)) ** 2
            grads_features_penalty = (1 - tf.norm(grads[1], axis=2)) ** 2
            return tf.reduce_mean(
                tf.reduce_mean(grads_adjacency_penalty, axis=(-2, -1))
                + tf.reduce_mean(grads_features_penalty, axis=(-1))
            )


    # Initialize the WGAN model
    wgan = GraphWGAN(generator, discriminator, discriminator_steps=1)

    wgan.compile(
        optimizer_generator=keras.optimizers.Adam(5e-4),
        optimizer_discriminator=keras.optimizers.Adam(5e-4),
    )

    #wgan.fit([adjacency_tensor, feature_tensor], epochs=epochs, batch_size=batch_size)

    st.subheader("Training the WGAN-GP Model")
    st.info("Training might take a few minutes depending on the dataset size and system performance.")

    progress_bar = st.progress(0)
    for epoch in range(epochs):
        wgan.fit([adjacency_tensor, feature_tensor], epochs=1, batch_size=batch_size)
        progress_bar.progress((epoch + 1) / epochs)
        st.write(f"Completed epoch {epoch + 1}/{epochs}")

    st.success("Training completed!")


    # Function to sample new molecules
    def sample(generator, batch_size):
        z = tf.random.normal((batch_size, LATENT_DIM))
        graph = generator.predict(z)
        # obtain one-hot encoded adjacency tensor
        adjacency = tf.argmax(graph[0], axis=1)
        adjacency = tf.one_hot(adjacency, depth=BOND_DIM, axis=1)
        # Remove potential self-loops from adjacency
        adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
        # obtain one-hot encoded feature tensor
        features = tf.argmax(graph[1], axis=2)
        features = tf.one_hot(features, depth=ATOM_DIM, axis=2)
        return [
            graph_to_molecule([adjacency[i].numpy(), features[i].numpy()])
            for i in range(batch_size)
        ]


    # Generate new molecules using the trained generator
    molecules = sample(wgan.generator, batch_size=48)
    st.subheader("Sample Novel Generated Molecules")

    # Display the generated molecules in a grid
    st.image(
        Draw.MolsToGridImage(
            [m for m in molecules if m is not None][:num_samples],
            molsPerRow=5,
            subImgSize=(150, 150),
            returnPNG=False
        )
    )
