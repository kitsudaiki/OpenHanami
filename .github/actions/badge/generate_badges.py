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
