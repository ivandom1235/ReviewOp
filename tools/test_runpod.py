import os
import sys
import asyncio
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual .env loading fallback
    env_file = root_path / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value.strip().strip('"').strip("'")

try:
    from runpod_flash import Endpoint
except ImportError:
    print("[!] 'runpod-flash' library not found. Please install it using: pip install runpod-flash")
    sys.exit(1)

async def run_connectivity_test():
    print("--- RunPod Flash Connectivity Test ---")
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_url = os.environ.get("RUNPOD_ENDPOINT_URL")
    
    if not api_key:
        print("[!] Warning: RUNPOD_API_KEY not found in environment.")
    
    # 1. Test basic environment setup
    print("[*] Verifying environment configuration...")
    if endpoint_url:
        print(f"[+] Endpoint URL found: {endpoint_url}")
    else:
        print("[!] Warning: RUNPOD_ENDPOINT_URL not found in environment.")

    # 2. Test the existing vLLM pod (Client Mode)
    if endpoint_url:
        if "api.runpod.ai/v2/" in endpoint_url:
            endpoint_id = endpoint_url.split("/v2/")[1].split("/")[0]
        else:
            endpoint_id = endpoint_url
            
        print(f"[*] Also testing connection to existing Pod ID: {endpoint_id}")
        vllm_ep = Endpoint(id=endpoint_id)
        try:
            # Requesting from the existing vLLM pod
            job = await vllm_ep.run({
                "prompt": "Reply with exactly one word: Success",
                "model_name": "llama3",
                "max_new_tokens": 10
            })
            
            print(f"[*] Job {job.id} submitted for existing Pod. Waiting...")
            # We use job.wait() instead of runsync for better diagnostics
            await job.wait(timeout=120)
            
            if job.output:
                print(f"[+] Existing Pod Response: {job.output}")
            elif job.error:
                print(f"[!] Existing Pod Error: {job.error}")
            else:
                print("[!] Existing Pod returned no output and no error.")
                
        except Exception as e:
            print(f"[!] Existing Pod connection failed: {e}")

if __name__ == "__main__":
    # Ensure RUNPOD_API_KEY is exportable for the SDK if not already
    if os.environ.get("RUNPOD_API_KEY"):
        os.environ["RUNPOD_API_KEY"] = os.environ["RUNPOD_API_KEY"]
        
    asyncio.run(run_connectivity_test())
