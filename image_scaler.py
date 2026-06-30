from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import StatePreparation
from PIL import Image
import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_qct_gate(n: int) -> QuantumCircuit:
    """
    Construit un circuit QCT-II sur n qubits de signal + 2 ancillas,
    selon la Figure 3 du papier (Ramos-Calderer 2022).

    Registre en entrée (n+2 qubits au total) :
        qubit 0       : ancilla a0  (Hadamard + contrôle CNOT)
        qubits 1..n   : signal q0..q_{n-1}
        qubit n+1     : ancilla a1  (porte X)

    Le circuit réalise :
        H  sur a0
        CNOT(a0 → q_i) pour tout i dans [0, n-1]
        X  sur a1
        QFT sur les n+2 qubits
    """
    total = n + 2
    qc = QuantumCircuit(total, name=f"QCT({n})")

    a0 = 0                      # ancilla MSB
    signal = list(range(1, n + 1))  # qubits du signal
    a1 = n + 1                  # ancilla LSB

    # Étape 1 : H sur l'ancilla de tête → copie symétrique du signal
    qc.h(a0)

    # Étape 2 : CNOT de a0 vers chaque qubit de signal
    for q in signal:
        qc.cx(a0, q)

    # Étape 3 : X sur l'ancilla de queue → décale sur les indices impairs
    qc.x(a1)

    # Étape 4 : QFT sur tout le système (n+2 qubits)
    qc.append(QFTGate(total), list(range(total)))

    return qc


def build_qct_inv_gate(n: int) -> QuantumCircuit:
    """QCT inverse = QCT† (on inverse l'ordre et on conjugue chaque porte)."""
    return build_qct_gate(n).inverse()


# ---------------------------------------------------------------------------
# Classe principale : interpolation QCT
# ---------------------------------------------------------------------------

class ImageGeneratorQCT:
    """
    Interpolation d'image par QCT (Quantum Cosine Transform) selon la
    Section III du papier « Efficient quantum interpolation of natural data »
    (Ramos-Calderer, 2022 — arXiv:2203.06196).

    Paramètres
    ----------
    image_path : str
        Chemin vers l'image source (RGB).
    ancilla_x : int
        Nombre de qubits ancillas ajoutés sur l'axe x (facteur d'agrandissement
        en x = 2^ancilla_x). Défaut : 1.
    ancilla_y : int
        Idem pour l'axe y. Défaut : 1.
    """

    def __init__(self, image_path: str, ancilla_x: int = 1, ancilla_y: int = 1):
        self.image = Image.open(image_path).convert("RGB")
        self.array = np.array(self.image)

        height, width, _ = self.array.shape
        self.nb_bits_x = math.ceil(math.log2(height))
        self.nb_bits_y = math.ceil(math.log2(width))
        
        self.ancilla_x = self.nb_bits_x + 1
        self.ancilla_y = self.nb_bits_y + 1

        # Taille de l'image de sortie
        self.out_size_x = self.nb_bits_x + ancilla_x   # bits → 2^(n+m) pixels en x
        self.out_size_y = self.nb_bits_y + ancilla_y

        self.qc = [None] * 3
        self.channel_sum = [None] * 3

        for ch in range(3):
            self.qc[ch] = self._build_channel_circuit(ch)

    # ------------------------------------------------------------------
    # Construction du circuit pour un canal couleur
    # ------------------------------------------------------------------

    def _encode_channel(self, ch: int) -> np.ndarray:
        """Encode un canal en vecteur d'état normalisé (amplitudes = sqrt(prob))."""
        gray = self.array[:, :, ch].astype(np.float64)
        height, width = gray.shape
        n_x, n_y = self.nb_bits_x, self.nb_bits_y

        total_sum = gray.sum()
        self.channel_sum[ch] = total_sum
        prob = gray / total_sum

        size = 2 ** (n_x + n_y)
        state = np.zeros(size)

        for x in range(height):
            for y in range(width):
                x_bits = format(x, f"0{n_x}b")
                y_bits = format(y, f"0{n_y}b")
                idx = int(x_bits + y_bits, 2)
                state[idx] = math.sqrt(prob[x, y])

        norm = np.linalg.norm(state)
        if norm > 0:
            state /= norm
        return state

    def _build_channel_circuit(self, ch: int) -> QuantumCircuit:
        """
        Construit le circuit QCT pour un canal.

        Architecture des qubits (de 0 à n_total-1) :

            [ancilla_x qubits] [ancilla_y qubits]   <- ancillas interpolation
            [2 ancillas QCT-x]                       <- ancillas QCT axe x
            [n_x qubits signal x]
            [2 ancillas QCT-y]                       <- ancillas QCT axe y
            [n_y qubits signal y]

        Soit :
            Section A (ancillas interpolation) : 0 .. ax+ay-1
            Section B (ancillas QCT-x)         : ax+ay .. ax+ay+1
            Section C (signal x)               : ax+ay+2 .. ax+ay+2+nx-1
            Section D (ancillas QCT-y)         : ax+ay+2+nx .. ax+ay+2+nx+1
            Section E (signal y)               : ax+ay+2+nx+2 .. fin
        """
        n_x = self.nb_bits_x
        n_y = self.nb_bits_y
        ax = self.ancilla_x
        ay = self.ancilla_y

        # ---- Indices des sections ----
        # Ancillas interpolation
        anc_interp_x = list(range(ax))                 # ax qubits
        anc_interp_y = list(range(ax, ax + ay))        # ay qubits

        # Bloc QCT-x  (ancilla a0_x, signal x, ancilla a1_x)
        base_x = ax + ay
        qct_x_a0 = base_x                              # ancilla MSB QCT-x
        sig_x = list(range(base_x + 1, base_x + 1 + n_x))  # signal x
        qct_x_a1 = base_x + 1 + n_x                   # ancilla LSB QCT-x

        # Bloc QCT-y  (ancilla a0_y, signal y, ancilla a1_y)
        base_y = base_x + 1 + n_x + 1
        qct_y_a0 = base_y
        sig_y = list(range(base_y + 1, base_y + 1 + n_y))
        qct_y_a1 = base_y + 1 + n_y

        n_total = base_y + 1 + n_y + 1
        n_classical = (n_x + ax) + (n_y + ay)  # bits mesurés en sortie

        qc = QuantumCircuit(n_total, n_classical)

        # ---- 1. Préparation de l'état (signal x + signal y) ----
        state = self._encode_channel(ch)
        prep_qubits = sig_x + sig_y          # ordre : x en MSB, y en LSB
        qc.append(StatePreparation(state), prep_qubits)

        # ---- 2. QCT sur l'axe x ----
        #   Applique QFT sur [qct_x_a0] + sig_x + [qct_x_a1]
        qct_x_qubits = [qct_x_a0] + sig_x + [qct_x_a1]
        qct_x = build_qct_gate(n_x)
        qc.append(qct_x, qct_x_qubits)

        # ---- 3. Interpolation sur l'axe x ----
        #   Swap les ancillas d'interpolation avec les MSB du signal x étendu,
        #   puis CNOT du qubit le plus significatif vers les ancillas.
        #
        #   Registre étendu x = anc_interp_x + [qct_x_a0] + sig_x
        #   (le a1 de QCT est le LSB, on l'ignore pour le padding)
        ext_x = anc_interp_x + [qct_x_a0] + sig_x   # taille ax + 1 + n_x

        # Swap ancillas interpolation ↔ premiers qubits du signal QCT
        for k in range(ax):
            qc.swap(ext_x[k], ext_x[ax + k])

        # CNOT du MSB étendu vers les ancillas d'interpolation
        msb_x = ext_x[ax]        # qubit le plus significatif après swap
        for k in range(ax):
            qc.cx(msb_x, ext_x[k])

        # ---- 4. QCT⁻¹ sur l'axe x (registre étendu) ----
        #   Registre QCT⁻¹ : anc_interp_x + [qct_x_a0] + sig_x + [qct_x_a1]
        full_x = anc_interp_x + [qct_x_a0] + sig_x + [qct_x_a1]
        qct_x_inv = build_qct_inv_gate(n_x + ax)
        qc.append(qct_x_inv, full_x)

        # ---- 5. Mesure de l'axe x (n_x + ax bits) ----
        meas_x = anc_interp_x + sig_x   
        out_x_bits = list(range(n_x + ax))
        qc.measure(meas_x, out_x_bits)

        # ---- 6. QCT sur l'axe y ----
        qct_y_qubits = [qct_y_a0] + sig_y + [qct_y_a1]
        qct_y = build_qct_gate(n_y)
        qc.append(qct_y, qct_y_qubits)

        # ---- 7. Interpolation sur l'axe y ----
        ext_y = anc_interp_y + [qct_y_a0] + sig_y

        for k in range(ay):
            qc.swap(ext_y[k], ext_y[ay + k])

        msb_y = ext_y[ay]
        for k in range(ay):
            qc.cx(msb_y, ext_y[k])

        # ---- 8. QCT⁻¹ sur l'axe y (registre étendu) ----
        full_y = anc_interp_y + [qct_y_a0] + sig_y + [qct_y_a1]
        qct_y_inv = build_qct_inv_gate(n_y + ay)
        qc.append(qct_y_inv, full_y)

        # ---- 9. Mesure de l'axe y ----
        meas_y = anc_interp_y + sig_y
        out_y_bits = list(range(n_x + ax, n_x + ax + n_y + ay))
        qc.measure(meas_y, out_y_bits)

        return qc

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def get_circuit(self) -> list:
        return self.qc

    def simulate(self, shots: int = 200_000) -> list:
        """
        Simule les 3 circuits (R, G, B) et reconstruit l'image interpolée.

        Retourne une liste de 3 tableaux numpy (uint8) correspondant aux
        canaux R, G, B de l'image agrandie.
        """
        sim = AerSimulator()
        self.data = []

        out_h = 2 ** self.out_size_x  # hauteur de l'image de sortie
        out_w = 2 ** self.out_size_y  # largeur de l'image de sortie
        n_bits_x = self.out_size_x
        n_bits_y = self.out_size_y

        for ch in range(3):
            job = sim.run(transpile(self.qc[ch], sim), shots=shots)
            counts = job.result().get_counts()

            table = np.zeros((out_h, out_w))

            for bitstring, count in counts.items():
                # Qiskit retourne le bitstring en ordre inverse (LSB à gauche)
                # Les bits classiques 0..(nx+ax-1) correspondent à l'axe x,
                # les bits (nx+ax)..(nx+ax+ny+ay-1) à l'axe y.
                # get_counts() concatène de MSB à LSB dans l'ordre classique.
                # On découpe :
                bits_x = bitstring[:n_bits_x]   # n_x + ax bits pour x
                bits_y = bitstring[n_bits_x:]   # n_y + ay bits pour y

                x = int(bits_x, 2)
                y = int(bits_y, 2)

                if x < out_h and y < out_w:
                    table[out_h - x - 1, out_w - y - 1] += count

            # Rééchellonnage vers les valeurs de pixel originales
            s = table.sum()
            if s > 0:
                table = table * (self.channel_sum[ch] / s)

            self.data.append(np.clip(table, 0, 255).astype(np.uint8))

        return self.data

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, save_path: str = None):
        rgb = np.stack([self.data[0], self.data[1], self.data[2]], axis=2)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        axes[0].imshow(self.array)
        axes[0].set_title(f"Original ({self.array.shape[1]}×{self.array.shape[0]})")
        axes[0].axis("off")

        axes[1].imshow(rgb)
        out_h, out_w = rgb.shape[:2]
        axes[1].set_title(f"QCT interpolé ({out_w}×{out_h})")
        axes[1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=150)
            print(f"Image sauvegardée : {save_path}")

        plt.show()


# ---------------------------------------------------------------------------
# Ancienne classe QFT (inchangée)
# ---------------------------------------------------------------------------

class ImageGenerator:
    """Interpolation QFT originale (Section II du papier)."""

    def __init__(self, image_path: str):
        self.image = Image.open(image_path).convert("RGB")
        self.array = np.array(self.image)

        height, width = self.array[:, :, 1].astype(np.float64).shape
        nb_bits_x = math.ceil(math.log2(height))
        nb_bits_y = math.ceil(math.log2(width))

        self.qc = [None] * 3
        self.sum = [None] * 3
        for i in range(3):
            gray = self.array[:, :, i].astype(np.float64)
            height, width = gray.shape
            prob = gray / gray.sum()
            self.sum[i] = gray.sum()
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

            size = 2 ** (nb_bits_x + nb_bits_y)
            state = np.zeros(size)
            for bitstring, p in zip(bit_data, probs_flat):
                idx = int(bitstring, 2)
                state[idx] = np.sqrt(p)
            state = state / np.linalg.norm(state)

            self.n_qubits = (2 * nb_bits_x) + (2 * nb_bits_y) - 2
            self.qc[i] = QuantumCircuit(self.n_qubits, self.n_qubits)
            self.size_x = nb_bits_x * 2 - 1
            self.size_y = nb_bits_y * 2 - 1
            ancilla_x = nb_bits_x - 1
            ancilla_y = nb_bits_y - 1
            start = ancilla_x + ancilla_y
            prepGate = StatePreparation(state)
            self.qc[i].append(prepGate, list(np.arange(start, self.n_qubits)))

            qft_a_x = QFTGate(nb_bits_x)
            qft_a_y = QFTGate(nb_bits_y)
            qft_a = [qft_a_x, qft_a_y]
            qft_b_x = QFTGate(nb_bits_x + ancilla_x)
            qft_b_y = QFTGate(nb_bits_y + ancilla_y)
            qft_b = [qft_b_x, qft_b_y]
            resolution = [nb_bits_x, nb_bits_y]
            ancilla = [ancilla_x, ancilla_y]
            for j in range(2):
                start_a = start + resolution[j] * j
                end_a = start_a + resolution[j]
                ancilla_tbl = np.arange(j * ancilla[j], j * ancilla[j] + ancilla[j])
                chunk = np.arange(start_a, end_a)
                tbl = list(ancilla_tbl)
                tbl.extend(chunk)
                self.qc[i].append(qft_a[j], tbl[-resolution[j]:])
                for k in range(ancilla[j]):
                    self.qc[i].swap(tbl[k], tbl[ancilla[j] + k])
                for k in range(resolution[j] - 1):
                    self.qc[i].cx(tbl[-1], tbl[ancilla[j] + k])
                self.qc[i].append(qft_b[j], tbl)
                self.qc[i].measure(
                    tbl,
                    list(np.arange(
                        j * (resolution[j] + ancilla[j]),
                        j * (resolution[j] + ancilla[j]) + resolution[j] + ancilla[j]
                    ))
                )

    def get_circuit(self):
        return self.qc

    def simulate(self, shots=200_000):
        sim = AerSimulator()
        self.data = []
        for channel in range(3):
            job = sim.run(transpile(self.qc[channel], sim), shots=shots)
            counts = job.result().get_counts()
            size_x = self.size_x
            size_y = self.size_y
            height = 2 ** size_x
            width = 2 ** size_y
            table = np.zeros((height, width))
            for bitstring, count in counts.items():
                x = int(bitstring[:size_x], 2)
                y = int(bitstring[size_y:], 2)
                table[height - x - 1, width - y - 1] = count
            table = table * table.sum() / self.sum[channel]
            self.data.append(table.astype(np.uint8))
        return self.data

    def plot(self, save_path: str = None):
        rgb = np.stack((self.data[0], self.data[1], self.data[2]), axis=2)
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.axis("off")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=150)
            print(f"Image sauvegardée : {save_path}")
        plt.show()


# ---------------------------------------------------------------------------
# Exemple d'utilisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- QFT (original) ---
    gen_qft = ImageGenerator(image_path="ref_32.png")
    qc = gen_qft.get_circuit()
    display(qc[0].draw(output="mpl"))
    # gen_qft.simulate()
    # gen_qft.plot("out_qft.png")

    # --- QCT (nouveau) ---
    # ancilla_x=1, ancilla_y=1 → image × 2 en chaque dimension
    # ancilla_x=2, ancilla_y=2 → image × 4 en chaque dimension
    gen_qct = ImageGeneratorQCT(
        image_path="ref_8.png",
        ancilla_x=1,
        ancilla_y=1,
    )
    qc = gen_qct.get_circuit()
    print(f"Nombre de qubits (canal R) : {qc[0].num_qubits}")
    display(qc[0].draw(output="mpl"))   # dans un notebook Jupyter

    #gen_qct.simulate(shots=200_000)
    #gen_qct.plot("out_qct.png")