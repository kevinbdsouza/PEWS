"""
Free configuration for Risk Atlas project - no paid APIs required.
"""
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class Settings:
    """Configuration settings for Risk Atlas project (free APIs only)."""
    
    # API Keys (only free ones)
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = "RiskAtlasBot/1.0"
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Comprehensive Subreddit Groups
    SUBREDDIT_GROUPS = {
        # Core AI / ML
        "core_ai": [
            "ArtificialIntelligence", "MachineLearning", "learnmachinelearning", "MLQuestions",
            "DeepLearning", "LanguageTechnology", "computervision", "datascience",
            "LocalLLaMA", "PromptEngineering", "OpenAI", "Anthropic", "StabilityAI",
            "Llama", "MistralAI", "QwenLM", "deepseek"
        ],

        # AGI / Alignment / Safety / Policy / Rationalist-adjacent
        "agi_alignment": [
            "agi", "singularity", "Futurology", "ControlProblem", "AIethics",
            "LessWrong", "TheMotte", "slatestarcodex", "EffectiveAltruism"
        ],

        # AI Companions / Parasocial with models
        "ai_companions": [
            "CharacterAI", "Replika", "AIcompanion", "AIgirlfriend", "AIBoyfriend"
        ],

        # Mental health (be careful & ethically handle)
        "mental_health": [
            "MentalHealth", "depression", "Anxiety", "SuicideWatch", "addiction", "BipolarReddit",
            "BPD", "OCD", "ADHD", "SocialAnxiety", "lonely", "ForeverAlone",
            "DecidingToBeBetter", "GetDisciplined", "productivity", "therapy", "psychotherapy",
            "CPTSD", "PTSD", "EatCheapAndHealthy"  # sometimes overlaps with coping/economic stress
        ],

        # Relationships / confessional spaces where AI-use signals show up
        "relationships_confessional": [
            "relationship_advice", "relationships", "TrueOffMyChest", "offmychest", "confession",
            "BreakUps", "AmItheAsshole"
        ],

        # AI art & creative tools (often includes dependence/offloading narratives)
        "ai_art": [
            "StableDiffusion", "midjourney", "AIGeneratedArt", "generativepy", "creativecoding"
        ],

        # Productivity / coding copilot reliance
        "dev_productivity": [
            "programming", "learnprogramming", "cscareerquestions", "datascience",
            "rust", "python", "golang", "reactjs", "cpp", "machinelearningmemes"
        ],

        # Digital well-being / tech overuse
        "digital_wellbeing": [
            "StopGaming", "NoSurf", "digitalminimalism", "InternetAddiction", "selfimprovement"
        ]
    }

    # Flattened subreddit list for data collection
    SUBREDDITS = sorted({s for grp in SUBREDDIT_GROUPS.values() for s in grp})
    
    # Model / vendor names (keep fresh)
    MODEL_KEYWORDS = [
        "GPT", "GPT-4", "GPT-4o", "GPT-4.1", "o1", "o3", "o3-mini",
        "Claude", "Claude 3", "Claude 3.5", "Sonnet", "Opus", "Haiku",
        "Gemini", "Gemini 1.5 Pro", "Gemini 1.5 Flash", "PaLM 2", "Gemma",
        "Llama", "Llama 3", "Llama 3.1", "Mistral", "Mixtral", "Command R",
        "Command R+", "Qwen", "RWKV", "Phi-3", "DeepSeek", "Jamba"
    ]

    # Generic LLM / assistant terms
    LLM_GENERAL = [
        "LLM", "large language model", "Artificial Intelligence", "chatbot", "AI assistant", "copilot",
        "system prompt", "chain of thought", "tool use",
        "agent", "AI agent", "reasoner"
    ]

    # Companion / parasocial
    COMPANION_KEYWORDS = [
        "AI companion", "AI friend", "AI girlfriend", "AI boyfriend",
        "virtual partner", "digital avatar", "parasocial", "simulated companion"
    ]

    # Cognitive offloading / dependence markers
    DEPENDENCE_MARKERS = [
        "can't think without", "rely on .* for everything", "I ask (ChatGPT|Claude|Gemini) for .*",
        "outsourcing my thinking", "depend on (it|them|the model)",
        "use it to make every decision", "I feel lost without", "AI told me to",
        "I trust (ChatGPT|Claude|Gemini) more than", "I need it to", "used it to decide"
    ]

    # Mental-health risk / symptom lexicon (trim/extend for precision vs recall)
    MENTAL_HEALTH_KEYWORDS = [
        "anxiety", "panic attack", "depression", "lonely", "isolation",
        "rumination", "intrusive thoughts", "derealization", "depersonalization",
        "paranoia", "compulsion", "addiction", "problematic use", "maladaptive",
        "suicidal", "self-harm", "hopeless", "burnout", "exhaustion",
        "catastrophizing", "OCD", "ADHD", "bipolar", "BPD", "PTSD", "CPTSD"
    ]

    # Ethics / governance / x-risk
    ETHICS_POLICY_KEYWORDS = [
        "AI ethics", "alignment", "AI safety", "x-risk", "AI governance",
        "RLHF", "RLAIF", "constitutional AI", "jailbreak", "prompt injection",
        "red teaming", "evals", "cardinality of harms", "anthropic principle (AI context)"
    ]

    # Productive coding reliance markers
    DEV_DEPENDENCE = [
        "copilot", "cursor", "code assistant", "pair programmer", "rubber duck",
        "I let it write my code", "it designs my architecture", "I can't debug without"
    ]

    # Comprehensive keyword list for filtering
    LLM_KEYWORDS = sorted(set(
        MODEL_KEYWORDS + LLM_GENERAL + COMPANION_KEYWORDS +
        DEPENDENCE_MARKERS + MENTAL_HEALTH_KEYWORDS +
        ETHICS_POLICY_KEYWORDS + DEV_DEPENDENCE
    ))
    
    # Time Period
    START_DATE = "2019-01-01"
    END_DATE = "2025-12-31"
    
    # Model Configuration
    QLORA_MODEL_NAME = "gpt2-medium"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # QLoRA Configuration (optimized for GPT-2)
    QLORA_R = 16
    QLORA_ALPHA = 32
    QLORA_DROPOUT = 0.1
    QLORA_LEARNING_RATE = 5e-4  # Higher learning rate for GPT-2
    QLORA_NUM_EPOCHS = 3
    QLORA_BATCH_SIZE = 8  # Larger batch size for GPT-2
    QLORA_GRADIENT_ACCUMULATION_STEPS = 2  # Fewer steps for GPT-2
    
    # Processing Parameters
    DEDUPLICATION_RADIUS = 0.08
    CONFIDENCE_THRESHOLD = 0.4
    TARGET_SAMPLE_SIZE = 10000
    SEED_LABEL_SIZE = 1000
    
    # Output Directories
    OUTPUT_BASE_DIR = "output/risk_atlas"
    MODEL_SAVE_DIR = "models"
    DATA_DIR = "data"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required API keys are present."""
        required_keys = [
            cls.REDDIT_CLIENT_ID,
            cls.REDDIT_CLIENT_SECRET,
            cls.GEMINI_API_KEY
        ]
        
        missing_keys = [key for key in required_keys if not key]
        if missing_keys:
            print(f"Missing required API keys: {missing_keys}")
            print("These are FREE APIs - no payment required!")
            print("Reddit: https://www.reddit.com/prefs/apps")
            print("Gemini: https://makersuite.google.com/app/apikey")
            return False
        return True
