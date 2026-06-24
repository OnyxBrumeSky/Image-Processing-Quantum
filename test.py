from PIL import Image
import numpy as np
import math

# =========================
# 1. CHARGEMENT IMAGE
# =========================
image = Image.open("ref_32.png").convert("RGB")
img_array = np.array(image)

# grayscale (simple canal R comme dans ton code)
gray = img_array[:, :, 0].astype(np.float64)

height, width = gray.shape

# =========================
# 2. PROBABILITÉ 2D
# =========================
prob = gray / gray.sum()   # somme = 1

# =========================
# 3. ENCODAGE BITSTRING
# =========================
nb_bits_x = math.ceil(math.log2(height))
nb_bits_y = math.ceil(math.log2(width))

bit_data = []
probs_flat = []

for x in range(height):
    for y in range(width):

        p = prob[x, y]

        x_bits = format(x, f"0{nb_bits_x}b")
        y_bits = format(y, f"0{nb_bits_y}b")

        bitstring = x_bits + y_bits

        bit_data.append(bitstring)
        probs_flat.append(p)

# =========================
# 4. VECTEUR QUANTIQUE
# =========================
size = 2 ** (nb_bits_x + nb_bits_y)

state = np.zeros(size)

for bitstring, p in zip(bit_data, probs_flat):
    idx = int(bitstring, 2)
    state[idx] = np.sqrt(p)   # IMPORTANT: amplitude

# normalisation quantum
state = state / np.linalg.norm(state)

# =========================
# 5. CIRCUIT QUANTIQUE
# =========================
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator

n_qubits = nb_bits_x + nb_bits_y

qc = QuantumCircuit(n_qubits, n_qubits)

# état quantique
state_prep = StatePreparation(state)
qc.append(state_prep, range(n_qubits))

qc.measure(range(n_qubits), range(n_qubits))
print(qc)
# =========================
# 6. SIMULATION
# =========================
simulator = AerSimulator()
shots = 50_000

transpiled = transpile(qc, simulator)
job = simulator.run(transpiled, shots=shots)
counts = job.result().get_counts()

# =========================
# 7. RECONSTRUCTION IMAGE
# =========================
table = np.zeros((height, width))

for bitstring, count in counts.items():

    bitstring = bitstring.replace(" ", "")

    x = int(bitstring[:nb_bits_x], 2)
    y = int(bitstring[nb_bits_y:], 2)
    
    table[x, y] = count / shots

# =========================
# 8. SAUVEGARDE IMAGE GRIS
# =========================
Image.fromarray(gray.astype(np.uint8)).save("gray.png")

# =========================
# 9. VISUALISATION BIOMES
# =========================
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_2d(noise_map: np.ndarray):
    fig, axes = plt.subplots(1, 1, figsize=(14, 5))

    axes[0].imshow(noise_map, cmap='gray')
    axes[0].set_title("Signal reconstruit")
    axes[0].axis('off')

    plt.tight_layout()
    plt.show()

plot_2d(table)







qc = QuantumCircuit(11, 10)

#qc.h(list(np.arange(4, 6)))
#qc.x(list(np.arange(4, 10)))



state_prep = StatePreparation(bit_data)
qc.append(state_prep, list(np.arange(4, 10)))

qft1a = QFTGate(3)
qft1b = QFTGate(3)

qc.append(qft1a, list(np.arange(4,7)))
qc.append(qft1b, list(np.arange(7,10)))

qc.swap(0, 7)
qc.swap(1, 8)

qc.cx(9,7)
qc.cx(9,8)

qc.swap(2, 4)
qc.swap(3, 5)

qc.cx(6,4)
qc.cx(6,5)

qft2a = QFTGate(5)
qft2b = QFTGate(5)



tbl = [0,1,7,8,9,2,3,4,5,6]

qc.append(qft2a, tbl[:5])
qc.append(qft2a, tbl[-5:])



qc.measure([0,1,7,8,9,2,3,4,5,6], list(np.arange(0,10)))

display(qc.draw(output="mpl"))


simulator = AerSimulator()
shots = 50_000

transpiled = transpile(qc, simulator)
job = simulator.run(transpiled, shots=shots)
counts = job.result().get_counts()




table = np.zeros((32,32))

for bitstring, count in counts.items():
    x = int(bitstring[:5],2)
    y = int(bitstring[-5:],2)
    table[x][y] = count / 50_000





def plot_2d(noise_map: np.ndarray):
    smoothed = noise_map

    # normalisation
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())

    # 7 seuils → 8 zones
    q = np.quantile(smoothed, [0.10, 0.22, 0.35, 0.50, 0.65, 0.78, 0.90])

    biome = np.digitize(smoothed, q)

    biome_colors = [
        "#2b4c7e",  # eau profonde
        "#4f8fbf",  # eau
        "#e6d3a3",  # plage
        "#a7d08c",  # prairie
        "#4f8b4a",  # forêt
        "#7c9a6d",  # collines
        "#8a8a8a",  # montagnes
        "#f2f2e6"   # neige
    ]

    biome_labels = [
        "Eau profonde",
        "Eau",
        "Plage",
        "Prairie",
        "Forêt",
        "Collines",
        "Montagnes",
        "Neige"
    ]

    cmap = ListedColormap(biome_colors)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].imshow(noise_map, cmap='gray')
    axes[0].set_title("Brut")
    axes[0].axis('off')



    im = axes[1].imshow(biome, cmap=cmap, vmin=0, vmax=7)
    axes[1].set_title("Biomes")
    axes[1].axis('off')

    patches = [plt.Rectangle((0, 0), 1, 1, color=biome_colors[i]) for i in range(8)]
    axes[1].legend(patches, biome_labels, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.show()

plot_2d(table)


