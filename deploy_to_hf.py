#!/usr/bin/env python
"""
Hugging Face Spaces Deployment Script
Uploads the symptom checker application to Hugging Face Spaces
"""

from huggingface_hub import upload_folder, HfApi
import os
import sys

print("=" * 70)
print("Hugging Face Spaces Deployment")
print("=" * 70)

# Check authentication
print("\n🔍 Checking Hugging Face authentication...")
try:
    api = HfApi()
    user_info = api.whoami()
    print(f"✅ Authenticated as: {user_info['name']}")
except Exception as e:
    print(f"❌ Authentication failed: {str(e)[:100]}")
    print("\n📝 To authenticate, run:")
    print("   huggingface-cli login")
    print("\n   Then paste your Hugging Face API token")
    print("   Get it at: https://huggingface.co/settings/tokens")
    sys.exit(1)

# List files to upload
print("\n📦 Files to upload:")
file_count = 0
for root, dirs, files in os.walk('.'):
    # Skip venv
    dirs[:] = [d for d in dirs if d != 'venv']
    for file in files:
        file_count += 1
        rel_path = os.path.join(root, file).replace('.\\', '').replace('\\', '/')
        size_mb = os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
        if size_mb > 1:
            print(f"   {rel_path:50} ({size_mb:.2f} MB)")
        else:
            print(f"   {rel_path:50}")

print(f"\n   Total files: {file_count}")

# Upload
print("\n⏳ Uploading to Hugging Face Spaces...")
print("   Repo: Sha-dab1/AI_medical_symptom_checker")
print("   Type: space")

try:
    result = upload_folder(
        folder_path='.',
        repo_id='Sha-dab1/AI_medical_symptom_checker',
        repo_type='space',
        commit_message='Deploy medical symptom checker with all accuracy fixes applied'
    )
    print(f"\n✅ Upload successful!")
    print(f"   Commit URL: {result}")
    print("\n🚀 Your app should be live at:")
    print("   https://huggingface.co/spaces/Sha-dab1/AI_medical_symptom_checker")
    print("\n   Note: It may take 1-2 minutes to build and start")
    
except Exception as e:
    print(f"\n❌ Upload failed!")
    print(f"   Error: {str(e)}")
    sys.exit(1)
