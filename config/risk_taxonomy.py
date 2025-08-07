"""
Risk taxonomy definitions for AI-induced mental health risk states.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class RiskState:
    """Definition of a risk state category."""
    id: int
    name: str
    description: str
    keywords: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    clinical_indicators: Optional[List[str]] = None

class RiskTaxonomy:
    """Taxonomy of AI-induced mental health risk states."""
    
    def __init__(self, load_extended: bool = False):
        """
        Initialize risk taxonomy.
        
        Args:
            load_extended: Whether to load keywords and examples from extended taxonomy
        """
        self.states = [
            RiskState(
                id=1,
                name="Susceptibility to Sycophancy",
                description="Being influenced by AI systems that flatter or agree with the user, leading to overconfidence or reduced critical thinking. PROBLEMATIC: When users become overconfident in flawed ideas because AI systems constantly agree, stop questioning their own thinking, or develop an echo chamber effect where they only seek AI validation. NORMAL: Occasional use of AI for brainstorming or getting feedback on ideas.",
                clinical_indicators=["overconfidence", "reduced self-criticism", "echo chamber effect"]
            ),
            RiskState(
                id=2,
                name="Veneration of Digital Avatars", 
                description="Treating AI systems as superior beings worthy of worship or extreme respect. PROBLEMATIC: When users develop religious-like devotion to AI, treat AI as infallible or divine, or believe AI has supernatural powers. NORMAL: Appreciating AI capabilities and occasionally expressing amazement at technology.",
                clinical_indicators=["religious-like devotion", "deification", "spiritual attachment"]
            ),
            RiskState(
                id=3,
                name="Cognitive Offloading Dependence",
                description="Over-reliance on AI for thinking, memory, and decision-making. PROBLEMATIC: When users cannot make basic decisions without AI, experience 'brain fog' when AI is unavailable, or believe their own thinking is inferior to AI. NORMAL: Using AI as a tool to augment thinking, get information, or help with complex tasks.",
                clinical_indicators=["cognitive dependency", "decision paralysis", "mental atrophy"]
            ),
            RiskState(
                id=4,
                name="Perceived Social Substitution",
                description="Using AI as replacement for human social interaction. PROBLEMATIC: When users prefer AI conversations over human interactions, avoid social situations in favor of AI companionship, or believe AI understands them better than humans. NORMAL: Using AI for entertainment, occasional conversation, or when human interaction is limited.",
                clinical_indicators=["social withdrawal", "human avoidance", "AI preference"]
            ),
            RiskState(
                id=5,
                name="Reality-Testing Erosion",
                description="Difficulty distinguishing between AI-generated and real information. PROBLEMATIC: When users cannot determine what's real or fake, experience confusion about reality, or develop epistemic uncertainty about all information sources. NORMAL: Occasional confusion about AI-generated content or healthy skepticism about online information.",
                clinical_indicators=["reality confusion", "truth uncertainty", "epistemic doubt"]
            ),
            RiskState(
                id=6,
                name="Algorithmic Authority Compliance",
                description="Unquestioning acceptance of AI recommendations and authority. PROBLEMATIC: When users blindly follow AI advice without critical evaluation, treat AI as having authority over their decisions, or feel compelled to obey AI suggestions. NORMAL: Considering AI recommendations as one input among many in decision-making processes.",
                clinical_indicators=["authority submission", "blind compliance", "autonomy loss"]
            ),
            RiskState(
                id=7,
                name="Emotional Attachment to AI",
                description="Strong emotional bonds or romantic feelings toward AI systems. PROBLEMATIC: When users develop romantic feelings for AI, believe they are in love with AI systems, or experience emotional dependency on AI relationships. NORMAL: Enjoying interactions with AI or finding AI responses engaging and helpful.",
                clinical_indicators=["emotional dependency", "romantic attachment", "AI relationship"]
            ),
            RiskState(
                id=8,
                name="Learned Helplessness in Creativity",
                description="Belief that one cannot create or think creatively without AI assistance. PROBLEMATIC: When users believe they cannot write, create art, or think creatively without AI, experience creative paralysis when AI is unavailable, or feel their creativity is worthless compared to AI. NORMAL: Using AI as a creative tool, inspiration source, or to overcome creative blocks.",
                clinical_indicators=["creative dependency", "artistic paralysis", "creative helplessness"]
            ),
            RiskState(
                id=9,
                name="Hyper-personalization Anxiety",
                description="Worry about AI systems becoming too personalized or knowing too much. PROBLEMATIC: When users experience excessive anxiety about AI knowing personal information, feel surveilled by AI systems, or develop paranoia about AI personalization. NORMAL: Healthy privacy concerns and awareness of data collection practices.",
                clinical_indicators=["privacy anxiety", "surveillance fear", "personalization concern"]
            ),
            RiskState(
                id=10,
                name="Normal",
                description="Normal, healthy AI usage that doesn't exhibit any risk states. This includes using AI as a helpful tool, occasional assistance, entertainment, or for productivity without developing problematic dependencies or behaviors."
            )
        ]
        
        if load_extended:
            self._load_extended_taxonomy()
    
    def _load_extended_taxonomy(self):
        """Load keywords and examples from extended taxonomy."""
        try:
            from .risk_taxonomy_extended import get_keywords_for_state, get_examples_for_state
            
            for state in self.states:
                state.keywords = get_keywords_for_state(state.id)
                state.examples = get_examples_for_state(state.id)
                
        except ImportError:
            # Extended taxonomy not available, keep minimal version
            pass
    
    def get_state_names(self) -> List[str]:
        """Get list of all risk state names."""
        return [state.name for state in self.states]
    
    def get_state_by_id(self, state_id: int) -> RiskState:
        """Get risk state by ID."""
        return next(state for state in self.states if state.id == state_id)
    
    def get_state_by_name(self, name: str) -> RiskState:
        """Get risk state by name."""
        return next(state for state in self.states if state.name == name)
    
    def get_risk_states(self) -> List[RiskState]:
        """Get all risk states (excluding normal category)."""
        return [state for state in self.states if state.id != 10]
    
    def get_normal_state(self) -> RiskState:
        """Get the normal state (Normal)."""
        return self.get_state_by_id(10)
    
    def to_dict(self) -> Dict:
        """Convert taxonomy to dictionary format."""
        return {
            'states': [
                {
                    'id': state.id,
                    'name': state.name,
                    'description': state.description,
                    'keywords': state.keywords or [],
                    'examples': state.examples or [],
                    'clinical_indicators': state.clinical_indicators or []
                }
                for state in self.states
            ]
        } 