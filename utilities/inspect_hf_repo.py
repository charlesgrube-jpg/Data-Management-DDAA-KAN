from huggingface_hub import HfApi

repo_id = "skypro1111/elevenlabs_dataset"
print(f"Inspecting {repo_id}...")
api = HfApi()
files = api.list_repo_files(repo_id, repo_type="dataset")
for f in files:
    print(f)
