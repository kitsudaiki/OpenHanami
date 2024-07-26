import os
import sys
import json

def main():
    category = sys.argv[1]
    status = json.loads(sys.argv[2])

    if not os.path.exists("./badges"):
        os.makedirs("./badges")
    if not os.path.exists(f"./badges/{category}"):
        os.makedirs(f"./badges/{category}")

    for e, state in status.items():
        path = f"./badges/{category}/{e}"
        if not os.path.exists(path):
            os.makedirs(path)

        e = e.replace("ubuntu-2204_clang-", "Ubuntu 22.04 Clang ")
        e = e.replace("ubuntu-2404_clang-", "Ubuntu 24.04 Clang ")
        e = e.replace("ubuntu-2204_gcc-", "Ubuntu 22.04 G++ ")
        e = e.replace("ubuntu-2404_gcc-", "Ubuntu 24.04 G++ ")
        e = e.replace("python-3_", "Python 3.")
        e = e.replace("kubernetes-1_", "Kubernetes 1.")
        
        ok = state == "success"
        data = {
            "schemaVersion": 1,
            "label": e,
            "message": "Passing" if ok else "Failing",
            "color": "brightgreen" if ok else "red"
        }
        
        with open(f"{path}/shields.json", "w") as file:
            json.dump(data, file)

if __name__ == "__main__":
    main()
