from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from qiskit.circuit.library import StatePreparation
from qiskit.visualization import plot_histogram
from PIL import Image
import math


class ImageGenerator:

    def __init__(
        self,
        image_path: str,
):
        self.image = Image.open(image_path).convert("RGB")
        self.array =  np.array(self.image)

        height, width = self.array[:, :, 1].astype(np.float64).shape
        nb_bits_x = math.ceil(math.log2(height))
        nb_bits_y = math.ceil(math.log2(width))
        
        self.qc = [None] * 3
        for i in range(3):
            gray = self.array[:, :, i].astype(np.float64)
            
            height, width = gray.shape
            
            prob = gray / gray.sum()
            
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
            self.qc[i] =  QuantumCircuit(self.n_qubits, self.n_qubits)

            self.size_x = nb_bits_x * 2 - 1
            self.size_y = nb_bits_y * 2 - 1
            
            ancilla_x = nb_bits_x - 1
            ancilla_y = nb_bits_y - 1
            start = ancilla_x + ancilla_y
            
            prepGate = StatePreparation(state)
            
            self.qc[i].append(
                prepGate, 
            list(np.arange(
                start, 
                self.n_qubits)))
            #self.qc[i].measure(np.arange(8,18),range(10))
                
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
                chunk = np.arange(start_a, end_a )
                tbl = list(ancilla_tbl)
                tbl.extend(chunk)
                self.qc[i].append(qft_a[j], tbl[-resolution[j]:])
    
                for k in range(ancilla[j]):
                    
                    self.qc[i].swap(tbl[k], tbl[ancilla[j] + k])
                for k in range(resolution[j] - 1):
                    
                    self.qc[i].cx(tbl[-1], tbl[ancilla[j] + k])
    
                self.qc[i].append(qft_b[j], tbl)
    
                self.qc[i].measure(tbl, list(np.arange(j * (resolution[j] + ancilla[j]), j * (resolution[j] + ancilla[j]) + resolution[j] + ancilla[j])))

    def get_circuit(self) -> QuantumCircuit: 
        return self.qc

    def simulate(self, shots=200_000):
    
        sim = AerSimulator()
        self.data = []
    
        for channel in range(3):
            job = sim.run(
                transpile(self.qc[channel], sim),
                shots=shots
            )
            counts = job.result().get_counts()
            size_x = self.size_x
            size_y = self.size_y

            height = 2**size_x
            width = 2**size_y
            
            table = np.zeros((height, width))
            
            for bitstring, count in counts.items():
                x = int(bitstring[:size_x], 2)
                y = int(bitstring[size_y:], 2)
                count = count * 100
                if count > 255:
                    table[height - x - 1, width - y - 1] = 255
                else :
                    table[height - x - 1, width - y - 1] = count
                    
                    
                
            #table = table * 255
            self.data.append(table.astype(np.uint8))
        return self.data
        
    def plot(self, save_path: str = None):

        rgb = np.stack(
            (
                self.data[0],  # R
                self.data[1],  # G
                self.data[2]   # B
            ),
            axis=2
        )
    
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.axis("off")
    
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=150)
            print(f"Image sauvegardée : {save_path}")
    
        plt.show()

# Exemple
if __name__ == "__main__":
    test = ImageGenerator(image_path="ref_32.png")
    qc = test.get_circuit()
    display(qc[1].draw(output="mpl"))
    test.simulate()
    test.plot("extended32.png")
