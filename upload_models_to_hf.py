import os
from huggingface_hub import HfApi

def upload_models():
    print("📰🔍 TruthLens Model Uploader for Hugging Face Hub")
    print("==================================================")
    print("This script will help you create a Hugging Face model repository")
    print("and upload all your trained model files securely.\n")
    
    # 1. Ask for Hugging Face repository ID
    repo_id = input("Enter your Hugging Face Repository ID (e.g., 'username/truthlens-models'): ").strip()
    if not repo_id:
        print("❌ Repository ID cannot be empty.")
        return
        
    # 2. Ask for Hugging Face Write Token
    print("You can get your token from: https://huggingface.co/settings/tokens")
    token = input("Enter your Hugging Face Access Token (with WRITE permission): ").strip()
    if not token:
        print("❌ Access token cannot be empty.")
        return
        
    api = HfApi()
    
    # 3. Create repo if it doesn't exist
    private_input = input("Should the Hugging Face repository be PRIVATE? (y/n, default is y): ").strip().lower()
    private = private_input != 'n'
    
    print(f"\nCreating/Verifying repository '{repo_id}' (Private: {private})...")
    try:
        api.create_repo(repo_id=repo_id, token=token, repo_type="model", private=private, exist_ok=True)
        print("✅ Repository ready.")
    except Exception as e:
        print(f"❌ Failed to verify/create repository: {e}")
        return
        
    # 4. Upload models folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    if not os.path.exists(models_dir):
        print(f"❌ Local models directory not found at: {models_dir}")
        return
        
    print(f"\nUploading contents of '{models_dir}' to '{repo_id}'...")
    try:
        # Upload the folder contents to the root of the Hugging Face repo
        api.upload_folder(
            folder_path=models_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print("\n🎉 ALL MODELS UPLOADED SUCCESSFULLY TO HUGGING FACE!")
        print(f"You can view your repository at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    upload_models()
