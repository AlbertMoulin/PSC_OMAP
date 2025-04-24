import os

L = []

for i in range(2,37):
    # check if file D:\Albert\Polytechnique\PSC\Code\code_papier_calvet_18_12\output\nln_CANFCPv2_FS10_' + i + '.txt exists
    if os.path.isfile(r'D:\Albert\Polytechnique\PSC\Code\code_papier_calvet_18_12\output\nln_CANFCPv2_FS10_' + str(i) + '.txt'):
        # read the file
        with open(r'D:\Albert\Polytechnique\PSC\Code\code_papier_calvet_18_12\output\nln_CANFCPv2_FS10_' + str(i) + '.txt', 'r') as file:
            # read the first line
            line = file.readline()
            L.append((i, float(line)))
            file.close()

print(min(L, key=lambda x: x[1]))
