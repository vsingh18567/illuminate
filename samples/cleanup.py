import os
import shutil

NON_DATA_FILES = {
    "indian_traffic": ["Indian Traffic Violations.csv", "prompt.txt"],
    "womens_shoes": [
        "Datafiniti_Womens_Shoes.csv",
        "prompt.txt",
    ],
}


def cleanup():
    for project_name, kept_files in NON_DATA_FILES.items():
        folder_files = os.listdir(project_name)
        for f in folder_files:
            if f not in kept_files:
                print(f"Removing {f} from {project_name}")
                if os.path.isfile(os.path.join(project_name, f)):
                    os.remove(os.path.join(project_name, f))
                elif os.path.isdir(os.path.join(project_name, f)):
                    shutil.rmtree(os.path.join(project_name, f))


if __name__ == "__main__":
    cleanup()
