import struct
import sys

L1_SIZE = 256
INPUT_SIZE = 768
L2_SIZE = 32
L3_SIZE = 32

def transpose_bin(in_path, out_path):
    with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
        # Magic and version
        fout.write(fin.read(8))
        
        # FT weights (INPUT_SIZE * L1_SIZE * 2 bytes)
        fout.write(fin.read(INPUT_SIZE * L1_SIZE * 2))
        
        # FT bias (L1_SIZE * 2 bytes)
        fout.write(fin.read(L1_SIZE * 2))
        
        # L2 weights (L1_SIZE*2 * L2_SIZE * 1 bytes)
        l2_raw = fin.read(L1_SIZE * 2 * L2_SIZE)
        # It's currently column-major: L1_SIZE*2 rows of L2_SIZE
        # We want row-major: L2_SIZE rows of L1_SIZE*2
        l2 = [[0 for _ in range(L2_SIZE)] for _ in range(L1_SIZE * 2)]
        idx = 0
        for i in range(L1_SIZE * 2):
            for j in range(L2_SIZE):
                l2[i][j] = l2_raw[idx]
                idx += 1
                
        for j in range(L2_SIZE):
            for i in range(L1_SIZE * 2):
                fout.write(struct.pack('<B', l2[i][j]))
                
        # L2 bias
        fout.write(fin.read(L2_SIZE * 4))
        
        # L3 weights (L2_SIZE * L3_SIZE * 1 bytes)
        l3_raw = fin.read(L2_SIZE * L3_SIZE)
        l3 = [[0 for _ in range(L3_SIZE)] for _ in range(L2_SIZE)]
        idx = 0
        for i in range(L2_SIZE):
            for j in range(L3_SIZE):
                l3[i][j] = l3_raw[idx]
                idx += 1
                
        for j in range(L3_SIZE):
            for i in range(L2_SIZE):
                fout.write(struct.pack('<B', l3[i][j]))
                
        # L3 bias
        fout.write(fin.read(L3_SIZE * 4))
        
        # Out weights
        fout.write(fin.read(L3_SIZE * 1))
        
        # Out bias
        fout.write(fin.read(4))

transpose_bin('nn.bin', 'nn_transposed.bin')
