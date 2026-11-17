import os

folders_and_files = os.listdir(".")
folders_unclean = [f for f in folders_and_files if os.path.isdir(f)]

skip = ["venv", "__pycache__"]
folders = [f for f in folders_unclean if f not in skip]

for folder in folders:
    print("carpeta:", folder)
    filename = folder + os.path.sep + "out.txt"
    csv = folder + os.path.sep + "out.csv"
    try:
        with open(filename, 'r') as f:
            whole = f.read()
    except FileNotFoundError:
        print(f"El archivo {filename} no existe, saltando")
        continue

    whole_lines = whole.split('\n')[1:]
    with open(csv, "w") as c:
        c.write("it,loss_phi,loss_G,HJB residual\n")
        for line in whole_lines:
            vals = line.split("|")
            vals = [v.strip() for v in vals]
            for val in vals:
                idx = vals.index(val)
                vals[idx] = val.split()
            assert len(vals) == 4
            writeline = f"{vals[0][-1]},{vals[1][-1]},{vals[2][-1]},{vals[3][-1]}\n"
            c.write(writeline)
