import streamlit as st
from rdkit import Chem, RDLogger
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import io  # Imported io module to capture model summaries
from rdkit.Chem import Draw, rdMolDescriptors, Crippen, PandasTools
from itertools import chain
from rdkit.Chem import FilterCatalog
from PIL import Image
import base64

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
expander_bar.markdown(
    """**Step 1:** In the User input-side panel, upload your input CSV file containing SMILES data. (*Example input file provided*)""")
expander_bar.markdown(
    """**Step 2:** In the "Select Atoms" section, choose the atoms you want to include in your analysis. (*Ensure that your SMILES strings in the uploaded CSV file contain only the atoms you selected in the "Select Atoms" section.*)""")
expander_bar.markdown("""**Step 3:** In the "Select Bonds" section, choose the types of chemical bonds to include.""")
expander_bar.markdown(
    """**Step 4:** In the "Set Parameters" section, adjust the hyperparameters for training the GAN.""")
expander_bar.markdown(
    """**Step 5:** Click the "✨ Generate Molecules" button in the sidebar to start the molecule generation process.""")

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
        optimizer_generator=tf.keras.optimizers.Adam(5e-4),
        optimizer_discriminator=tf.keras.optimizers.Adam(5e-4),
    )

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
        # Obtain one-hot encoded adjacency tensor
        adjacency = tf.argmax(graph[0], axis=1)
        adjacency = tf.one_hot(adjacency, depth=BOND_DIM, axis=1)
        # Remove potential self-loops from adjacency
        adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
        # Obtain one-hot encoded feature tensor
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

    # Dictionary of functional group SMARTS patterns
    acceptable_groups = {
        # Acceptable functional groups
        "amine": 'C(N)',  # Generally accepted; can form hydrogen bonds.
        "Aromatic": 'c1ccccc1',  # Provides planarity; useful in interactions.
        "Aliphatic cyclic": 'C1CCCCC1',  # Provides rigidity; beneficial in drug design.
        "Nitrile": 'C#N',  # Stable and can enhance biological activity.
        "Phenol": 'c1ccccc1O',  # Commonly used; can participate in hydrogen bonding.
        "Sulfonic acid": 'S(=O)(=O)(O)',  # Stable; can improve solubility and bioactivity.
        "Thioether": 'C-S-C',  # Usually acceptable; can exhibit varying reactivity.
        "Sulfide": 'C-S-C',  # Usually acceptable; can exhibit varying reactivity.
        "Urea": 'C(=O)N(N)',  # Stable; used in various pharmaceutical compounds.
        "Sulfamide": 'S(=O)(=O)N',  # Stable and can be used in drug design.
        "Cyclopropane": 'C1CC1',  # Provides ring strain; can enhance reactivity.
        "Cyclobutane": 'C1CCC1',  # Provides rigidity; useful in drug design.
        "Pyridine": 'c1ccncc1',  # Heterocyclic compound; often used in drug design for its properties.
        "Thiazole": 'c1sccn1',  # Heterocyclic; known for bioactivity in various compounds.
        "Imidazole": 'c1ncccn1',  # Heterocyclic; important in drug design.
        "Furan": 'c1ccoc1',  # Heterocyclic; sometimes used in pharmaceuticals.
        "Oxime": 'C(=N-O)',  # Useful in drug synthesis.
        "Triazole": 'c1nnn1',  # Important in drug design; has diverse applications.
        "Carboxylic acid": 'C(=O)O',
        # Acceptable; common in drug design for functionality and interaction with biological targets.
        "Aliphatic long chain": 'C*',
        # Generally acceptable; long chains can enhance lipid solubility, aiding in absorption.
        "Alcohol": 'C(O)',  # Generally acceptable; widely used in pharmaceuticals for solubility and reactivity.
        "Ketone": 'C(=O)C',  # Generally acceptable; common functional group that can participate in hydrogen bonding.
        "Ester": 'C(=O)O[C]',  # Generally acceptable; used in prodrugs for improved bioavailability.
        "Amine": 'C(N)',  # Generally acceptable; essential for many drugs due to their role in receptor interactions.
        "Ether": 'C-O-C',  # Generally acceptable; often used in pharmaceuticals for their stability and low reactivity.
        "Fluoride": 'C(F)',  # Individual halides; generally acceptable; can enhance bioactivity through fluorination.
        "Chloride": 'C(Cl)',
        # Individual halides; generally acceptable; can influence lipophilicity and biological activity.
        "Bromide": 'C(Br)',  # Individual halides; generally acceptable; used to modulate drug properties.
        "Iodide": 'C(I)',
        # Individual halides; generally acceptable; iodinated compounds can have specific therapeutic applications.
        "Phosphate": 'P(=O)(O)(O)O',
        # Generally acceptable; crucial in biological processes and drug design (e.g., nucleotides).
        "Amide": 'C(=O)N',
        # Generally acceptable; prevalent in drug design for its stability and ability to form hydrogen bonds.
        "Quaternary ammonium": '[N+]',
        # Generally acceptable; used in drug design for enhanced solubility and transport properties.
        "Isolated alkene": 'C=C',
        # Generally acceptable; common in organic compounds; can be involved in various reactions.
        "2-halo pyridine": 'n1ccccc1[X]',
        # Where X = F, Cl, Br, I; generally acceptable; used for specific biological interactions.
        "Oxygen-nitrogen single bond": 'O-N',
        # Generally acceptable; can enhance hydrogen bonding and molecular interactions.
        "Quaternary nitrogen": '[N+](C)(C)(C)(C)',  # Generally acceptable; often used in drug design for solubility.
        "Triple bond": 'C#C',
        # Generally acceptable; common in organic compounds, influencing reactivity and stability.
        "Phosphor": 'P',
        # Generally acceptable; context-dependent; phosphorous compounds can have therapeutic applications.
        "Diketo group": 'C(=O)C(=O)',  # Generally acceptable; can participate in chelation and molecular interactions.
        "Three-membered heterocycle": '[r3]',
        # Generally acceptable; often found in pharmaceuticals for diverse activities.
    }

    # Dictionary of functional group SMARTS patterns
    acceptable_with_caution_groups = {
        # Sometimes acceptable (requires caution)
        "Hydroxylamine": 'N(O)=C',  # Can be unstable and reactive; should be approached with caution.
        "Carbamate": 'C(=O)N(C)C',  # Stable but should be monitored for potential toxicity.
        "Hydrazone": 'C=N-N',  # May have undesirable reactivity; caution advised.
        "Alkyne": 'C#C',  # Highly reactive; generally requires caution.
        "Aldehyde": 'C(=O)[H]',  # Reactive; can lead to side reactions.
        "Secondary amine": 'C(N)[C]',  # Generally acceptable; requires caution due to potential reactivity.
        "Tertiary amine": 'C(N)(N)',  # Generally acceptable; can participate in hydrogen bonding but may be reactive.
        "Anhydride": 'C(=O)OC(=O)',  # Can be reactive; should be used cautiously due to hydrolysis risks.
        "Dithiol": 'C(S)(S)',  # Can form reactive species; caution is advised.
        "Tetrahydrofuran": 'C1CCOC1',  # Generally stable; commonly used but should be monitored.
        "imine": 'C=NC',  # Generally acceptable; can form stable structures but may be reactive in certain conditions.
        "Phenol": 'c1ccccc1O',  # Can be toxic in high concentrations; serves as a crucial component in many drugs.
        "Thiol": 'C(S)',
        # Generally acceptable; although some thiols can be toxic, many have essential biological roles.
        "Alkyl halide": 'C[X]',
        # Where X = F, Cl, Br, I; generally acceptable but requires caution due to potential toxicity.
        "Iodine": 'C(I)',  # Individual halides; generally acceptable with caution; iodine is essential in some drugs.
        "Charged oxygen or sulfur atoms": '[O-]|[S-]',
        # Generally acceptable; can participate in nucleophilic reactions but may be reactive.
        "Het-C-het not in ring": 'C[!C][!C]',  # Generally acceptable; context-dependent; can be stable or reactive.
        "N oxide": 'N(=O)[O-]',  # Generally acceptable; depends on context; can be involved in biological processes.
        "Halogenated ring": 'c1ccccc1[X]',
        # Where X = F, Cl, Br, I; generally acceptable but requires caution due to reactivity.
        "Thioester": 'C(=S)OC',  # Generally acceptable; context-dependent; can have significant biological roles.
    }

    # Dictionary of functional group SMARTS patterns
    toxic_groups = {
        # Toxic functional groups
        "Chlorinated compounds": 'C(Cl)',
        # Potentially toxic; can lead to organochlorine accumulation in biological systems.
        "Brominated compounds": 'C(Br)',  # Potentially toxic; can exhibit endocrine disruption properties.
        "Acyl halide": 'C(=O)Cl',  # Highly reactive; should be avoided due to reactivity and potential side effects.
        "Benzyl alcohol": 'C(C1=CC=CC=C1)(O)',  # Can be metabolized to toxic metabolites.
        "Thiocarbonyl group": 'C(=S)C',
        # Potentially toxic; can exhibit reactive properties, affecting biological systems.
        "Aniline": 'c1ccc(N)cc1',  # Can be mutagenic; requires caution due to potential carcinogenicity and toxicity.
        "Acid halide": 'C(=O)X',
        # Where X = F, Cl, Br, I; highly reactive; should be avoided due to reactivity and potential side effects.
        "Iodinated compounds": 'C(I)',
        # Generally avoided due to associated toxicity; used cautiously in specific applications.
        "Hydrazine": 'N=N',  # Highly reactive and toxic; should be avoided due to instability and health risks.
        "Nitro group": 'N(=O)(=O)C',  # Often toxic; should be avoided due to potential for carcinogenicity.
        "Cyanide": 'C#N',  # Highly toxic; should be avoided in drug design due to safety concerns.
        "Diazo group": 'N=N',  # Highly reactive; should be avoided due to stability issues and potential explosiveness.
        "Isocyanate": 'C(=O)N=',  # Highly reactive; significant safety risks; should be avoided in drug design.
        "Heavy metal": '[Pb]|[Hg]|[Cd]',  # Highly toxic; should be avoided in drug design due to severe health risks.
        "Azide": 'N=N=N',  # Highly explosive; should be avoided in drug design due to instability.
        "Peroxide": 'O-O',  # Highly reactive; generally avoided due to potential for decomposition and toxicity.
        "Benzidine": 'c1ccc(N=N)c2ccccc2',  # Known carcinogen; should be avoided due to significant health risks.
        "Chlorinated hydrocarbons": 'C(Cl)(C)(C)',
        # Toxic; associated with various health risks and environmental concerns.
        "Dimethylformamide (DMF)": 'C(N(C)C)(C)=O',
        # Toxic solvent; should be used with caution due to safety concerns.
        "Selenium compounds": 'Se',
        # Some can be toxic; should be avoided in drug design unless specific activity is desired.
        "Formamide": 'C(=O)N',  # Potentially toxic; should be used cautiously due to associated health risks.
        "Dioxin": 'C1=C(C2=C(C=C1)O)C(=O)C=C2',  # Highly toxic; should be avoided due to carcinogenic properties.
        "Vinyl chloride": 'C=C(Cl)',  # Known carcinogen; should be avoided due to health risks.
        "Nitrosamines": 'N(=O)(C)',  # Carcinogenic; should be avoided due to significant safety concerns.
        "Aromatic amine": 'c1ccc(N)cc1',  # Can be mutagenic; requires caution due to potential for adverse effects.
        "Triazene": 'N=N-N',
        # Sometimes acceptable; can be used in certain drug designs but may be reactive and require careful handling.
        "Aromatic hydroxylamine": 'c1cc(N(O)C)cc1',
        # Sometimes acceptable; useful in drug design for specific interactions, but can be toxic and reactive under certain conditions.

        ##doi: 10.1007/978-1-4939-7899-1_13
        "1,2-Dicarbonyls": '[C;X3](=O)([C;X3](=O))',  # Generally avoided; can be reactive and lead to side effects.
        "Acyl Halides": '[F,Cl,Br,I][C;X3]=[O,S]',
        # Highly reactive; typically avoided due to instability and potential toxicity.
        "Aldehydes": '[#6][C;H1]=[O;X1]',  # Sometimes acceptable; useful in synthesis, but can be toxic and reactive.
        "Alkyl Halides, P/S Halides, Mustards": '[Cl,Br,I][P,S,C&H2&X4,C&H3&X4]',
        # Generally avoided; potential for reactivity and toxicity.
        "Alkyl Sulfonates, Sulfate Esters": '[#6]O[S;X4](=O)=O',
        # Sometimes acceptable; can serve as leaving groups, but may be reactive.
        "Alpha-halocarbonyls": '[#6][C;X3](=[O;X1])-[C;H1,H2]-[F,Cl,Br,I]',
        # Generally avoided; high reactivity and potential toxicity.
        "Alpha-Beta Unsaturated Nitriles": '[#6]=[#6]C#N',
        # Sometimes acceptable; can be useful in certain drug designs, but caution required.
        "Anhydride": '[O;X2]([CX3,S,P]=O)([CX3,S,P]=O)',  # Generally avoided; reactive and may cause side effects.
        "Azides": '[#6][N;X2]=[N;X2]=[N;X1]',  # Generally avoided; highly explosive and reactive.
        "Beta-Carbonyl Quaternary Nitrogen": '[C;X3](=O)[C][N,n;X4]',
        # Sometimes acceptable; useful in certain contexts but requires caution.
        "Beta-Heterosubstituted Carbonyls": '[O;X1]=C[C;H2]C[F,Cl,Br,I]',
        # Generally avoided; may lead to instability and toxicity.
        "Carbodiimides": '[#6][N;X2]=[C;X2]=[N;X2][#6]',
        # Sometimes acceptable; useful in peptide synthesis but can be reactive.
        "Diazos, Diazoniums": '[#6]~[NX2;+0,+1]~[NX1;−1,+1,+0]',
        # Generally avoided; high reactivity and stability issues.
        "Disulfides": '[S;X2]~[S;X2]',
        # Sometimes acceptable; can be beneficial in drug design, but may cause stability issues.
        "Epoxides, Thioepoxides, Aziridines": '[O,N,S;X2;r3](C)C',
        # Sometimes acceptable; useful in synthetic chemistry but can be reactive.
        "Formates": '[O;X2][C;H1]=O',  # Sometimes acceptable; can serve as intermediates in synthesis.
        "Halopyrimidines": '[F,Cl,Br,I]c(nc)nc',
        # Sometimes acceptable; used in some drug designs, but caution is advised.
        "Heteroatom-Heteroatom Single Bonds": '[O,N,S;X2]~[O,N,S;X2]',
        # Generally acceptable; often found in bioactive compounds.
        "Imines (Schiff’s Base)": '[N;X2]([!#1])=[C;X3][C;H2,H3]',
        # Sometimes acceptable; can be useful for specific interactions in drug design.
        "Isocyanates, Isothiocyanates": '[#6][N;X2]=C=[O,S&X1]',  # Generally avoided; highly reactive and can be toxic.
        "Isonitriles": '[#6][N;X2]#[C;X1]',
        # Sometimes acceptable; can be used in specific cases, but requires caution.
        "Michael Acceptors": '[#6]=[#6][#6,#16]=[O]',
        # Sometimes acceptable; useful in certain drug design contexts, but may be reactive.
        "Nitroaromatic": '[c;X3][$([NX3](=O)=O),$([NX3+](=O)[O−])]',
        # Generally avoided; can be toxic and carcinogenic.
        "Nitrosos, Nitrosamines": '[#6,#7][N;X2](=O)',  # Generally avoided; highly toxic and carcinogenic.
        "Perhalomethylketones": '[#6][C;X3](=O)[C;X4]([F,Cl,Br,I])([F,Cl,Br,I])[F,Cl,Br,I]',
        # Generally avoided; highly reactive and toxic.
        "Phosphines, Phosphoranes": '[#6][#15&X3,#15&X5]([#6])~[#6]',
        # Sometimes acceptable; can be used in synthesis but may have stability issues.
        "Phosphinyl Halides": '[P;X3][Cl,Br,I]',  # Generally avoided; can be reactive and lead to side effects.
        "Reactive Cyanides": 'N#C[C&X4,C&X3]~[O&X1,O&H1&X2]',  # Highly toxic; should be avoided in drug design.
        "Sulfenyl, Sulfinyl, Sulfonyl Halides": '[F,Cl,Br,I][$([SX2]),$([S;X3]=O),$([S;X4](=O)=O)]',
        # Generally avoided; potential for reactivity and toxicity.
        "Thiocyanate": '[#6][S]C#N',  # Generally avoided; highly toxic and should be avoided in drug design.
        "Thioesters": '*SC(=O)*',
        # Sometimes acceptable; can be useful in specific contexts, but may have stability issues.
        "Thiourea": '[SX1]~C([N&!R&X3,N&!R&X2])[N&!R&X3,N&!R&X2]',
        # Sometimes acceptable; can be beneficial in drug design but requires caution.
        "Vinyl Halides": '[C;X2]=[C;X2]-[F,Cl,Br,I]',  # Generally avoided; potential for reactivity and toxicity.
    }


    # Function to check for functional groups and categorize them
    def check_functional_groups(mol):
        matched_groups = {
            'acceptable': [],
            'acceptable_with_caution': [],
            'toxic': []
        }

        functional_groups = {**acceptable_groups, **acceptable_with_caution_groups, **toxic_groups}

        for group_name, pattern in functional_groups.items():
            if pattern is None:
                continue  # Skip this iteration if pattern is None

            # Convert SMARTS pattern to a RDKit Mol object
            smarts_mol = Chem.MolFromSmarts(pattern)
            if smarts_mol is None:
                print(f"Warning: Invalid SMARTS pattern for {group_name}")
                continue  # Skip invalid SMARTS patterns

            # Check for substructure match
            if mol.HasSubstructMatch(smarts_mol):
                if group_name in acceptable_groups:  # Define acceptable_groups based on your criteria
                    matched_groups['acceptable'].append(group_name)
                elif group_name in acceptable_with_caution_groups:  # Define sometimes_acceptable_groups
                    matched_groups['acceptable_with_caution'].append(group_name)
                else:  # Assume toxic if it doesn't fit the above
                    matched_groups['toxic'].append(group_name)

        return matched_groups


    # Generate SMILES from generated molecules
    generated_smiles = [Chem.MolToSmiles(m) for m in molecules if m]

    # Create DataFrame with SMILES
    maindf = pd.DataFrame({'smiles': generated_smiles})
    st.info(f'Total compounds before removing duplicates: {len(maindf)}')

    # Add molecule column
    PandasTools.AddMoleculeColumnToFrame(maindf, 'smiles', 'molecule')

    # Calculate molecular descriptors
    maindf['rot_bonds'] = maindf.molecule.apply(rdMolDescriptors.CalcNumRotatableBonds)
    maindf['logP'] = maindf.molecule.apply(Crippen.MolLogP)
    maindf['HBD'] = maindf.molecule.apply(rdMolDescriptors.CalcNumHBD)
    maindf['HBA'] = maindf.molecule.apply(rdMolDescriptors.CalcNumHBA)
    maindf['mw'] = maindf.molecule.apply(rdMolDescriptors.CalcExactMolWt)
    maindf['inchi'] = maindf.molecule.apply(Chem.MolToInchiKey)

    # Remove duplicates
    maindf.drop_duplicates(subset='inchi', inplace=True)
    st.success(f'Total compounds after removing duplicates: {len(maindf)}')

    # Load PAINS filters
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog = FilterCatalog.FilterCatalog(params)

    # Check PAINS alerts for each molecule
    maindf['pains_alert'] = None  # Create a new column for PAINS alerts

    for i, row in maindf.iterrows():
        mol = row['molecule']  # Access the molecule from the DataFrame
        if mol is not None:
            entry = catalog.GetFirstMatch(mol)
            if entry:
                maindf.at[i, 'pains_alert'] = entry.GetDescription()  # Store PAINS alert description
            else:
                maindf.at[i, 'pains_alert'] = "No PAINS alert"

    # Display PAINS alerts
    pains_filtered = maindf[maindf['pains_alert'] != "No PAINS alert"]
    if not pains_filtered.empty:
        st.info("PAINS (Pan-Assay Interference Compounds) Alerts found:")
        st.dataframe(pains_filtered[['smiles', 'pains_alert']])
    else:
        st.info("No PAINS (Pan-Assay Interference Compounds) alerts found.")

    # Remove molecules with PAINS alerts
    pains_free_df = maindf[maindf['pains_alert'] == "No PAINS alert"]

    # Display total compounds after removing PAINS alerts
    st.success(f'Total compounds after removing PAINS alerts: {len(pains_free_df)}')


    # Categorize functional groups
    def categorize_functional_groups(smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        return check_functional_groups(mol)


    # Create separate columns for each category of functional groups in the filtered dataframe
    pains_free_df['acceptable'] = pains_free_df['smiles'].apply(
        lambda sm: ', '.join(check_functional_groups(Chem.MolFromSmiles(sm))['acceptable']))
    pains_free_df['acceptable_with_caution'] = pains_free_df['smiles'].apply(
        lambda sm: ', '.join(check_functional_groups(Chem.MolFromSmiles(sm))['acceptable_with_caution']))
    pains_free_df['toxic'] = pains_free_df['smiles'].apply(
        lambda sm: ', '.join(check_functional_groups(Chem.MolFromSmiles(sm))['toxic']))

    # Add a summary column based on functional group categories
    pains_free_df['classification'] = pains_free_df.apply(
        lambda row: 'Pharmacophoric' if row['acceptable'] and not row['toxic'] else
        ('Toxicophoric' if row['toxic'] and (row['acceptable'] or row['acceptable_with_caution']) else
         ('Sometimes Acceptable' if row['acceptable_with_caution'] and not (
                 row['acceptable'] or row['toxic']) else 'no_match')),
        axis=1
    )

    # Display the functional group categorization
    st.subheader("Classification of Virtual Library of Novel Molecules based on their Functional Groups")


    # st.dataframe(pains_free_df[['smiles', 'classification', 'acceptable', 'acceptable_with_caution', 'toxic']])

    # Function to generate molecule images
    def get_molecular_structure(mol):
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=(150, 150))  # Create an image of the molecule
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')  # Save to bytes
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()  # Return bytes for the image


    # Add molecule image column to the DataFrame
    pains_free_df['molecular_structure'] = pains_free_df['molecule'].apply(lambda x: get_molecular_structure(x))


    # Display filtered DataFrame with images in Streamlit
    def display_dataframe_with_images(df, download_key):
        # Create a new DataFrame for display with 'molecular_structure' first
        display_df = df[
            ['molecular_structure', 'smiles', 'classification', 'acceptable', 'acceptable_with_caution',
             'toxic']].copy()

        # Convert images to a format that can be displayed in Streamlit
        display_df['molecular_structure'] = display_df['molecular_structure'].apply(
            lambda
                img: f'<img src="data:image/png;base64,{base64.b64encode(img).decode()}" width="150" height="150"/>' if img is not None else ""
        )

        # Use st.markdown to render the DataFrame as HTML
        st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Prepare the DataFrame for CSV export (removing image column)
        csv_df1 = df[['smiles', 'classification', 'acceptable', 'acceptable_with_caution', 'toxic']]

        # Create CSV string
        csv = csv_df1.to_csv(index=False)

        # Download link
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='classified_data.csv',
            mime='text/csv',
            key=download_key  # Unique key for the download button
        )


    # Call the function to display the DataFrame with unique keys
    display_dataframe_with_images(pains_free_df, download_key="download_button_1")

    # Flatten and count occurrences of functional groups
    acceptable_groups_flat_list = [group for sublist in pains_free_df['acceptable'].str.split(', ') for group in sublist
                                   if group]
    acceptable_with_caution_flat_list = [group for sublist in pains_free_df['acceptable_with_caution'].str.split(', ')
                                         for group in sublist if group]
    toxic_groups_flat_list = [group for sublist in pains_free_df['toxic'].str.split(', ') for group in sublist if group]

    # Count the occurrences of each functional group
    acceptable_groups_counts = pd.Series(acceptable_groups_flat_list).value_counts().reset_index()
    acceptable_with_caution_counts = pd.Series(acceptable_with_caution_flat_list).value_counts().reset_index()
    toxic_groups_counts = pd.Series(toxic_groups_flat_list).value_counts().reset_index()

    acceptable_groups_counts.columns = ['Pharmacophoric', 'count']
    acceptable_with_caution_counts.columns = ['Toxicophoric', 'count']
    toxic_groups_counts.columns = ['Sometimes Acceptable', 'count']

    # Display functional group counts using columns
    st.subheader("Functional Groups Count")
    row1_1, row1_2, row1_3 = st.columns([1, 1, 1])  # Equal width for each column

    # Display each dataframe in a separate column
    with row1_1:
        st.warning("Pharmacophoric Groups")
        st.dataframe(acceptable_groups_counts.head(), use_container_width=True)  # Show only the head

    with row1_2:
        st.warning("Toxicophoric Groups")
        st.dataframe(acceptable_with_caution_counts.head(), use_container_width=True)

    with row1_3:
        st.warning("Sometimes Acceptable Groups")
        st.dataframe(toxic_groups_counts.head(), use_container_width=True)

    # Apply custom filters
    filtered_df = pains_free_df[
        (pains_free_df['rot_bonds'] <= 3) &
        (pains_free_df['logP'] <= 5) &
        (pains_free_df['HBD'] <= 5) &
        (pains_free_df['HBA'] <= 10) &
        (pains_free_df['mw'] <= 500) &
        (pains_free_df['classification'] == 'Pharmacophoric')
        ]


    # Visualize all ring systems in the generated structures
    def find_ring_systems(mol):
        if mol is None:
            return []
        ring_info = mol.GetRingInfo()
        ring_systems = []
        for ring in ring_info.AtomRings():
            ring_smiles = Chem.MolFragmentToSmiles(mol, ring)
            ring_systems.append(ring_smiles)
        return list(set(ring_systems))  # Return unique ring systems


    # Apply the ring system finder
    filtered_df['ring_systems'] = filtered_df['molecule'].apply(find_ring_systems)

    # Flatten the list of ring systems into a single list for counting
    ring_list = list(chain(*filtered_df.ring_systems.values))
    ring_series = pd.Series(ring_list)
    ring_df = pd.DataFrame(ring_series.value_counts()).reset_index()
    ring_df.columns = ["SMILES", "Count"]

    # Display ring system counts
    st.subheader("Ring System Counts:")
    st.dataframe(ring_df)

    # Visualize filtered molecules
    filtered_mols = [m for m in filtered_df['molecule'] if m is not None]

    # Display filtered results
    st.subheader("Virtually Screened Molecular Data")
    st.success(f'Final Total Pharmacophoric Compounds after Screening with Lipinskis Rule of Five : {len(filtered_df)}')

    # Add ring system information to DataFrame
    maindf['ring_systems'] = maindf.molecule.apply(find_ring_systems)

    # Assuming 'filtered_df' is your DataFrame and 'molecule' column contains RDKit molecule objects
    filtered_df['Molecular_structure'] = filtered_df['molecule'].apply(lambda x: get_molecular_structure(x))


    # Display filtered DataFrame with images in Streamlit
    def display_dataframe_with_images(df):
        # Create a new DataFrame for display with 'molecular_structure' first
        display_df = df[
            ['Molecular_structure', 'smiles', 'rot_bonds', 'logP', 'HBD', 'HBA', 'mw', 'inchi',
             'classification', 'ring_systems']].copy()

        # Convert images to a format that can be displayed in Streamlit
        display_df['Molecular_structure'] = display_df['Molecular_structure'].apply(
            lambda
                img: f'<img src="data:image/png;base64,{base64.b64encode(img).decode()}" width="150" height="150"/>' if img is not None else ""
        )

        # Use st.markdown to render the DataFrame as HTML
        st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Prepare the DataFrame for CSV export (removing image column)
        csv_df = df[['smiles', 'rot_bonds', 'logP', 'HBD', 'HBA', 'mw', 'inchi', 'classification',
                     'ring_systems']]

        # Create CSV string
        csv2 = csv_df.to_csv(index=False)

        # Download link with a unique key
        st.download_button(
            label="Download data as CSV",
            data=csv2,
            file_name='screened_data.csv',
            mime='text/csv',
        )


    # Display filtered DataFrame with images and a unique key
    display_dataframe_with_images(filtered_df)
