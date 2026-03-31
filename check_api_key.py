# Diagnostic script to check Gemini API key and list available models

import google.generativeai as genai
import os

def check_api_key(api_key=None):
    """Check API key and list available models"""
    
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("\n💡 To check your API key, run one of these:")
        print("   PowerShell: python check_api_key.py -k YOUR_API_KEY")
        print("   Or set env var first: $env:GEMINI_API_KEY='your-key'; python check_api_key.py")
        return
    
    print(f"✅ API Key found (preview): {api_key[:15]}...{api_key[-5:]}")
    print("\n🔍 Checking API key validity...\n")
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        print("✅ API configuration successful\n")
        
        # List all available models
        print("📋 Fetching available models...\n")
        models = genai.list_models()
        
        # Filter for text generation models
        text_models = []
        embedding_models = []
        
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                text_models.append(model.name)
            elif 'embedContent' in model.supported_generation_methods:
                embedding_models.append(model.name)
        
        print("=" * 80)
        print("📝 TEXT GENERATION MODELS (for RAG responses):")
        print("=" * 80)
        if text_models:
            for model_name in sorted(text_models):
                print(f"  ✅ {model_name}")
        else:
            print("  ❌ No text generation models available")
        
        print("\n" + "=" * 80)
        print("🔢 EMBEDDING MODELS (for vector indexing):")
        print("=" * 80)
        if embedding_models:
            for model_name in sorted(embedding_models):
                print(f"  ✅ {model_name}")
        else:
            print("  ❌ No embedding models available")
        
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATIONS:")
        print("=" * 80)
        
        if text_models:
            # Find the best available model
            preferred_models = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.5-pro']
            for preferred in preferred_models:
                if any(preferred in m for m in text_models):
                    print(f"  ✅ Use '{preferred}' for text generation")
                    break
            else:
                print(f"  ⚠️  Use '{text_models[0]}' for text generation")
        else:
            print("  ❌ No suitable text generation model found")
        
        if embedding_models:
            if any('embedding-001' in m for m in embedding_models):
                print("  ✅ Use 'models/embedding-001' for embeddings")
            else:
                print(f"  ⚠️  Use '{embedding_models[0]}' for embeddings")
        else:
            print("  ❌ No suitable embedding model found")
        
        print("\n" + "=" * 80)
        print("⚠️  API KEY LIMITATIONS:")
        print("=" * 80)
        print("  • Free tier API keys have rate limits (requests per minute/day)")
        print("  • Some models may require paid tier or specific permissions")
        print("  • Check Google AI Studio for your quota limits")
        print("  • If no models appear, your API key may be invalid or expired")
        
    except Exception as e:
        print(f"❌ Error checking API key: {str(e)}")
        print("\n💡 Possible issues:")
        print("  • Invalid API key")
        print("  • API key expired")
        print("  • Network connectivity issue")
        print("  • API not enabled for your project")

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
    
    check_api_key(api_key)
