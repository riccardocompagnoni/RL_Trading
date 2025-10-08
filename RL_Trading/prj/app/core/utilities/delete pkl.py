import os, glob

if __name__ == "__main__":
    #folder_path = "C:/Users/Riccardo/OneDrive - Politecnico di Milano/Webeep/Thesis/RL_Trading/results"
    folder_path = "/home/restelli/a2a/RL_Trading/results"
    [os.remove(f) for f in glob.glob(os.path.join(folder_path, "**", "*.pkl"), recursive=True)]