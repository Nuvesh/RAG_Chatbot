# Diagnostic script to check Cohere API key and available models

import os

def check_cohere_api_key(api_key=None):
    """Check Cohere API key and test connectivity"""
    
    if not api_key:
        api_key = os.getenv('COHERE_API_KEY')
    
    if not api_key:
        print("❌ Error: COHERE_API_KEY environment variable not set")
        print("\n💡 To check your API key, run one of these:")
        print("   PowerShell: python check_cohere_key.py -k YOUR_API_KEY")
        print("   Or set env var first: $env:COHERE_API_KEY='your-key'; python check_cohere_key.py")
        return
    
    print(f"✅ API Key found (preview): {api_key[:10]}...{api_key[-5:]}")
    print("\n🔍 Checking API key validity...\n")
    
    try:
        # Try to import cohere
        import cohere
        print("✅ Cohere library imported successfully\n")
        
        # Initialize client
        client = cohere.Client(api_key)
        print("✅ Client initialized\n")
        
        # Test embedding model
        print("📊 Testing embedding model (embed-english-v3.0)...")
        try:
            response = client.embed(
                texts=["This is a test"],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            # Access embeddings from the response
            # response.embeddings can be either a list of lists or an object with float_ attribute
            if isinstance(response.embeddings, list):
                embeddings_list = response.embeddings
            else:
                embeddings_list = getattr(response.embeddings, 'float_', None)
            
            if embeddings_list and len(embeddings_list) > 0:
                print(f"✅ Embedding successful! Vector dimension: {len(embeddings_list[0])}")
            else:
                print("❌ No embeddings returned")
        except Exception as e:
            print(f"❌ Embedding test failed: {str(e)}")
        
        print("\n" + "=" * 80)
        print("🤖 Testing LLM models...")
        print("=" * 80)
        
        # Test LLM models
        models_to_test = [
            ("command-r-plus", "Most capable model"),
            ("command-r", "Fast and capable"),
            ("command", "Legacy model"),
        ]
        
        for model_name, description in models_to_test:
            try:
                print(f"\nTesting {model_name} ({description})...")
                response = client.chat(
                    message="Say 'Hello, I am working!' if you can read this",
                    model=model_name,
                    temperature=0.1,
                    max_tokens=20
                )
                print(f"✅ {model_name} works! Response: {response.text[:50]}...")
            except Exception as e:
                print(f"❌ {model_name} failed: {str(e)}")
        
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATIONS:")
        print("=" * 80)
        print("  ✅ Use 'embed-english-v3.0' for embeddings (dimension: 1024)")
        print("  ✅ Use 'command-r-plus' for best RAG performance")
        print("  ✅ Use 'command-r' for faster, cheaper queries")
        
        print("\n" + "=" * 80)
        print("⚠️  API KEY LIMITATIONS:")
        print("=" * 80)
        print("  • Free tier: Limited to certain rate limits (requests/minute)")
        print("  • Check Cohere dashboard for your specific quota")
        print("  • Some enterprise models may require upgraded access")
        print("  • Token limits vary by model")
        
    except ImportError:
        print("❌ Error: Cohere library not installed")
        print("\n💡 Install it using: pip install cohere")
    except Exception as e:
        print(f"❌ Error checking API key: {str(e)}")
        print("\n💡 Possible issues:")
        print("  • Invalid API key")
        print("  • API key expired")
        print("  • Network connectivity issue")
        print("  • API not enabled for your account")

if __name__ == "__main__":
    import sys
    
    # Check for command line argument
    api_key = None
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-k', '--key']:
            if len(sys.argv) > 2:
                api_key = sys.argv[2]
            else:
                print("❌ Error: No API key provided after -k flag")
                sys.exit(1)
        else:
            api_key = sys.argv[1]
    
    check_cohere_api_key(api_key)
