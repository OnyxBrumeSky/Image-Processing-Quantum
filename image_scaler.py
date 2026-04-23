from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from qiskit.circuit.library import StatePreparation
from distribution_generator import DistributionGenerator
from qiskit.visualization import plot_histogram
from PIL import Image


class PerlinGenerator:

    def __init__(
        self,
        resolution : (int, int) = (2, 2), 
        image_path: str,
        shots : int = 50_000,
):
        self.image = Image.open(image_path).convert("RGB")
        self.array =  np.array(self.image)
        self.shots = shots
        
        for i in range(3):
            gray = self.array[:, :, 1].astype(np.float64)
            
            height, width = gray.shape
            
            prob = gray / gray.sum()

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

            size = 2 ** (nb_bits_x + nb_bits_y)
            
            state = np.zeros(size)
            
            for bitstring, p in zip(bit_data, probs_flat):
                idx = int(bitstring, 2)
                state[idx] = np.sqrt(p)

            state = state / np.linalg.norm(state)
            
            self.n_qubits = 2 * nb_bits_x + 2 * nb_bits_y - 2
            self.qc[i] =  QuantumCircuit(self.n_qubits, self.n_qubits)
            
            ancilla_x = nb_bits_x - 1
            ancilla_y = nb_bits_y - 1
            start = ancilla_x + ancilla_y
            
            prepGate = StatePreparation(state)
            
            self.qc.append(
                prepGate, 
            list(np.arange(
                start, 
                self.n_qubits)))
            
            qft_a_x = QFTGate(nb_bits_x)
            qft_a_y = QFTGate(nb_bits_y)
            
            qft_b_x = QFTGate(nb_bits_x + ancilla_x)
            qft_b_y = QFTGate(nb_bits_y + ancilla_y)

        for i in range(n_dimmensions):
            
            start_a = start + i * resolution
            end_a = start_a + resolution
            ancilla_tbl = np.arange(i * ancilla, i * ancilla + ancilla)
            chunk = np.arange(start_a, end_a )
            tbl = list(ancilla_tbl)
            tbl.extend(chunk)
            self.qc.append(qft_a, tbl[-resolution:])

            for j in range(ancilla):
                self.qc.swap(tbl[j], tbl[ancilla + j])
            for j in range(resolution - 1):
                self.qc.cx(tbl[-1], tbl[ancilla + j])

            self.qc.append(qft_b, tbl)

            self.qc.measure(tbl, list(np.arange(i * self.chunk_size, i * self.chunk_size + self.chunk_size)))

    def get_circuit(self) -> QuantumCircuit: 
        return self.qc

    def simulate(self, shots: int = 50_000) -> dict:
        sim = AerSimulator()
        job = sim.run(transpile(self.qc, sim), shots=shots)
        counts = job.result().get_counts()
        self.counts = {k: v / shots for k, v in sorted(counts.items())}
        size = 2**(self.chunk_size)

        if self.n_dims == 1 : # Simple distribution
            self.data = self.counts
            
        elif self.n_dims == 2 : # 2D map
            table = np.zeros((size,size))
            for bitstring, count in counts.items():
                x = int(bitstring[:self.chunk_size],2)
                y = int(bitstring[-self.chunk_size:],2)
                table[x][y] = count
            self.data = (table - table.min()) / (table.max() - table.min())
        elif self.n_dims == 3:
            vol = np.zeros((size, size, size))
        
            for bitstring, count in counts.items():
                x = int(bitstring[:self.chunk_size], 2) % size
                y = int(bitstring[self.chunk_size:2*self.chunk_size], 2) % size
                z = int(bitstring[-self.chunk_size:], 2) % size
                vol[x, y, z] = count
        
            self.data = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
            
        return self.counts


# Exemple
if __name__ == "__main__":
    test = PerlinGenerator(n_dimmensions=3, resolution=3, distrib="random")
    qc = test.get_circuit()
    display(qc.draw(output="mpl"))
    test.simulate()
    test.plot()
