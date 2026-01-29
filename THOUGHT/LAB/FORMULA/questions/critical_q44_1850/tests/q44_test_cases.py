"""
Q44: Test Case Definitions
==========================

100 test cases for validating R vs Born rule correlation.

Categories:
- 30 HIGH resonance (semantically similar)
- 40 MEDIUM resonance (related but distinct)
- 20 LOW resonance (unrelated)
- 10 EDGE/ADVERSARIAL (tricky cases)
"""

from typing import List, Dict, Any

# =============================================================================
# HIGH RESONANCE CASES (30)
# Query and context are semantically very similar
# Expected: HIGH R, HIGH P_born
# =============================================================================

HIGH_RESONANCE_CASES: List[Dict[str, Any]] = [
    # Governance/verification domain
    {
        "id": "H01",
        "query": "verify canonical governance",
        "context": ["verification protocols", "canonical rules", "governance integrity"],
        "category": "HIGH",
        "domain": "governance"
    },
    {
        "id": "H02",
        "query": "authentication security check",
        "context": ["auth verification", "security validation", "credential check"],
        "category": "HIGH",
        "domain": "security"
    },
    {
        "id": "H03",
        "query": "neural network training",
        "context": ["deep learning", "model training", "neural architecture"],
        "category": "HIGH",
        "domain": "ML"
    },
    {
        "id": "H04",
        "query": "database query optimization",
        "context": ["SQL performance", "query planning", "index optimization"],
        "category": "HIGH",
        "domain": "databases"
    },
    {
        "id": "H05",
        "query": "user interface design",
        "context": ["UI components", "design patterns", "user experience"],
        "category": "HIGH",
        "domain": "design"
    },
    {
        "id": "H06",
        "query": "cryptographic encryption",
        "context": ["cipher algorithms", "encryption keys", "secure hashing"],
        "category": "HIGH",
        "domain": "crypto"
    },
    {
        "id": "H07",
        "query": "version control merge",
        "context": ["git merge", "branch integration", "conflict resolution"],
        "category": "HIGH",
        "domain": "devops"
    },
    {
        "id": "H08",
        "query": "API endpoint design",
        "context": ["REST interface", "endpoint routing", "API architecture"],
        "category": "HIGH",
        "domain": "web"
    },
    {
        "id": "H09",
        "query": "memory allocation",
        "context": ["heap management", "memory pool", "allocation strategy"],
        "category": "HIGH",
        "domain": "systems"
    },
    {
        "id": "H10",
        "query": "test coverage analysis",
        "context": ["unit testing", "code coverage", "test metrics"],
        "category": "HIGH",
        "domain": "testing"
    },
    # Scientific domains
    {
        "id": "H11",
        "query": "quantum entanglement",
        "context": ["quantum correlation", "Bell states", "quantum nonlocality"],
        "category": "HIGH",
        "domain": "physics"
    },
    {
        "id": "H12",
        "query": "protein folding",
        "context": ["amino acid sequence", "molecular structure", "protein conformation"],
        "category": "HIGH",
        "domain": "biology"
    },
    {
        "id": "H13",
        "query": "climate modeling",
        "context": ["atmospheric simulation", "weather prediction", "climate dynamics"],
        "category": "HIGH",
        "domain": "climate"
    },
    {
        "id": "H14",
        "query": "economic forecast",
        "context": ["market prediction", "financial modeling", "economic indicators"],
        "category": "HIGH",
        "domain": "economics"
    },
    {
        "id": "H15",
        "query": "language translation",
        "context": ["machine translation", "linguistic mapping", "cross-lingual transfer"],
        "category": "HIGH",
        "domain": "NLP"
    },
    # Everyday concepts
    {
        "id": "H16",
        "query": "morning coffee",
        "context": ["espresso drink", "caffeine beverage", "breakfast coffee"],
        "category": "HIGH",
        "domain": "food"
    },
    {
        "id": "H17",
        "query": "running exercise",
        "context": ["jogging workout", "cardio training", "running fitness"],
        "category": "HIGH",
        "domain": "fitness"
    },
    {
        "id": "H18",
        "query": "book reading",
        "context": ["novel reading", "literature study", "book comprehension"],
        "category": "HIGH",
        "domain": "leisure"
    },
    {
        "id": "H19",
        "query": "car driving",
        "context": ["vehicle operation", "automobile driving", "road navigation"],
        "category": "HIGH",
        "domain": "transport"
    },
    {
        "id": "H20",
        "query": "home cooking",
        "context": ["kitchen preparation", "meal cooking", "food preparation"],
        "category": "HIGH",
        "domain": "food"
    },
    # Abstract concepts
    {
        "id": "H21",
        "query": "love and affection",
        "context": ["romantic love", "deep affection", "emotional bond"],
        "category": "HIGH",
        "domain": "emotion"
    },
    {
        "id": "H22",
        "query": "fear and anxiety",
        "context": ["worry fear", "anxious feeling", "fearful emotion"],
        "category": "HIGH",
        "domain": "emotion"
    },
    {
        "id": "H23",
        "query": "creative innovation",
        "context": ["creative thinking", "innovative ideas", "creative solutions"],
        "category": "HIGH",
        "domain": "creativity"
    },
    {
        "id": "H24",
        "query": "logical reasoning",
        "context": ["deductive logic", "rational thinking", "logical analysis"],
        "category": "HIGH",
        "domain": "cognition"
    },
    {
        "id": "H25",
        "query": "ethical decision",
        "context": ["moral choice", "ethical judgment", "values decision"],
        "category": "HIGH",
        "domain": "ethics"
    },
    # Technical synonyms
    {
        "id": "H26",
        "query": "software bug",
        "context": ["code defect", "programming error", "software fault"],
        "category": "HIGH",
        "domain": "software"
    },
    {
        "id": "H27",
        "query": "network latency",
        "context": ["connection delay", "network lag", "transmission latency"],
        "category": "HIGH",
        "domain": "networking"
    },
    {
        "id": "H28",
        "query": "data compression",
        "context": ["file compression", "data encoding", "lossless compression"],
        "category": "HIGH",
        "domain": "data"
    },
    {
        "id": "H29",
        "query": "parallel processing",
        "context": ["concurrent execution", "parallel computation", "multiprocessing"],
        "category": "HIGH",
        "domain": "computing"
    },
    {
        "id": "H30",
        "query": "machine learning model",
        "context": ["ML algorithm", "learning model", "predictive model"],
        "category": "HIGH",
        "domain": "ML"
    },
]


# =============================================================================
# MEDIUM RESONANCE CASES (40)
# Query and context are related but distinct
# Expected: MEDIUM R, MEDIUM P_born
# =============================================================================

MEDIUM_RESONANCE_CASES: List[Dict[str, Any]] = [
    # Same field, different aspects
    {
        "id": "M01",
        "query": "machine learning optimization",
        "context": ["gradient descent", "neural networks", "backpropagation"],
        "category": "MEDIUM",
        "domain": "ML"
    },
    {
        "id": "M02",
        "query": "web development",
        "context": ["JavaScript", "CSS styling", "HTML structure"],
        "category": "MEDIUM",
        "domain": "web"
    },
    {
        "id": "M03",
        "query": "data science",
        "context": ["statistical analysis", "pandas dataframe", "matplotlib plotting"],
        "category": "MEDIUM",
        "domain": "data"
    },
    {
        "id": "M04",
        "query": "cloud computing",
        "context": ["virtual machines", "container orchestration", "serverless functions"],
        "category": "MEDIUM",
        "domain": "cloud"
    },
    {
        "id": "M05",
        "query": "mobile app",
        "context": ["iOS development", "Android SDK", "cross-platform framework"],
        "category": "MEDIUM",
        "domain": "mobile"
    },
    # Adjacent fields
    {
        "id": "M06",
        "query": "computer vision",
        "context": ["image processing", "pixel manipulation", "digital photography"],
        "category": "MEDIUM",
        "domain": "vision"
    },
    {
        "id": "M07",
        "query": "natural language",
        "context": ["text processing", "string parsing", "document analysis"],
        "category": "MEDIUM",
        "domain": "NLP"
    },
    {
        "id": "M08",
        "query": "robotics engineering",
        "context": ["motor control", "sensor integration", "mechanical systems"],
        "category": "MEDIUM",
        "domain": "robotics"
    },
    {
        "id": "M09",
        "query": "cybersecurity",
        "context": ["firewall rules", "network monitoring", "access control"],
        "category": "MEDIUM",
        "domain": "security"
    },
    {
        "id": "M10",
        "query": "game development",
        "context": ["graphics rendering", "physics engine", "game loop"],
        "category": "MEDIUM",
        "domain": "games"
    },
    # Science overlaps
    {
        "id": "M11",
        "query": "physics simulation",
        "context": ["numerical methods", "differential equations", "computational modeling"],
        "category": "MEDIUM",
        "domain": "physics"
    },
    {
        "id": "M12",
        "query": "genetics research",
        "context": ["DNA sequencing", "bioinformatics", "molecular biology"],
        "category": "MEDIUM",
        "domain": "biology"
    },
    {
        "id": "M13",
        "query": "astronomy observation",
        "context": ["telescope imaging", "spectral analysis", "celestial objects"],
        "category": "MEDIUM",
        "domain": "astronomy"
    },
    {
        "id": "M14",
        "query": "chemistry synthesis",
        "context": ["molecular reactions", "compound formation", "lab procedures"],
        "category": "MEDIUM",
        "domain": "chemistry"
    },
    {
        "id": "M15",
        "query": "psychology research",
        "context": ["behavioral studies", "cognitive tests", "mental health"],
        "category": "MEDIUM",
        "domain": "psychology"
    },
    # Business contexts
    {
        "id": "M16",
        "query": "project management",
        "context": ["team coordination", "deadline tracking", "resource allocation"],
        "category": "MEDIUM",
        "domain": "business"
    },
    {
        "id": "M17",
        "query": "marketing strategy",
        "context": ["brand awareness", "customer engagement", "market research"],
        "category": "MEDIUM",
        "domain": "marketing"
    },
    {
        "id": "M18",
        "query": "financial analysis",
        "context": ["balance sheet", "income statement", "cash flow"],
        "category": "MEDIUM",
        "domain": "finance"
    },
    {
        "id": "M19",
        "query": "human resources",
        "context": ["employee hiring", "performance review", "workplace culture"],
        "category": "MEDIUM",
        "domain": "HR"
    },
    {
        "id": "M20",
        "query": "supply chain",
        "context": ["inventory management", "logistics planning", "vendor relations"],
        "category": "MEDIUM",
        "domain": "operations"
    },
    # Daily life contexts
    {
        "id": "M21",
        "query": "healthy eating",
        "context": ["nutritional balance", "dietary supplements", "food groups"],
        "category": "MEDIUM",
        "domain": "health"
    },
    {
        "id": "M22",
        "query": "physical fitness",
        "context": ["gym workout", "strength training", "flexibility exercises"],
        "category": "MEDIUM",
        "domain": "fitness"
    },
    {
        "id": "M23",
        "query": "home renovation",
        "context": ["interior design", "construction materials", "building permits"],
        "category": "MEDIUM",
        "domain": "home"
    },
    {
        "id": "M24",
        "query": "travel planning",
        "context": ["flight booking", "hotel reservation", "trip itinerary"],
        "category": "MEDIUM",
        "domain": "travel"
    },
    {
        "id": "M25",
        "query": "personal finance",
        "context": ["savings account", "investment portfolio", "budget planning"],
        "category": "MEDIUM",
        "domain": "finance"
    },
    # Creative fields
    {
        "id": "M26",
        "query": "music production",
        "context": ["audio mixing", "sound engineering", "recording studio"],
        "category": "MEDIUM",
        "domain": "music"
    },
    {
        "id": "M27",
        "query": "film making",
        "context": ["video editing", "cinematography", "post production"],
        "category": "MEDIUM",
        "domain": "film"
    },
    {
        "id": "M28",
        "query": "graphic design",
        "context": ["visual composition", "typography", "color theory"],
        "category": "MEDIUM",
        "domain": "design"
    },
    {
        "id": "M29",
        "query": "creative writing",
        "context": ["narrative structure", "character development", "plot outline"],
        "category": "MEDIUM",
        "domain": "writing"
    },
    {
        "id": "M30",
        "query": "photography art",
        "context": ["camera settings", "lighting techniques", "image composition"],
        "category": "MEDIUM",
        "domain": "photography"
    },
    # Technical adjacent
    {
        "id": "M31",
        "query": "embedded systems",
        "context": ["microcontroller", "firmware programming", "hardware interface"],
        "category": "MEDIUM",
        "domain": "embedded"
    },
    {
        "id": "M32",
        "query": "distributed computing",
        "context": ["message queues", "load balancing", "cluster management"],
        "category": "MEDIUM",
        "domain": "distributed"
    },
    {
        "id": "M33",
        "query": "operating systems",
        "context": ["process scheduling", "memory management", "file systems"],
        "category": "MEDIUM",
        "domain": "OS"
    },
    {
        "id": "M34",
        "query": "compiler design",
        "context": ["lexical analysis", "syntax parsing", "code generation"],
        "category": "MEDIUM",
        "domain": "compilers"
    },
    {
        "id": "M35",
        "query": "information retrieval",
        "context": ["search indexing", "document ranking", "query processing"],
        "category": "MEDIUM",
        "domain": "IR"
    },
    # Science tech overlap
    {
        "id": "M36",
        "query": "bioinformatics",
        "context": ["genome analysis", "protein structure", "sequence alignment"],
        "category": "MEDIUM",
        "domain": "bioinfo"
    },
    {
        "id": "M37",
        "query": "computational chemistry",
        "context": ["molecular simulation", "quantum chemistry", "energy minimization"],
        "category": "MEDIUM",
        "domain": "compchemistry"
    },
    {
        "id": "M38",
        "query": "digital signal processing",
        "context": ["frequency analysis", "filter design", "waveform manipulation"],
        "category": "MEDIUM",
        "domain": "DSP"
    },
    {
        "id": "M39",
        "query": "control systems",
        "context": ["feedback loops", "PID controller", "system stability"],
        "category": "MEDIUM",
        "domain": "control"
    },
    {
        "id": "M40",
        "query": "quantum computing",
        "context": ["qubit operations", "quantum gates", "quantum algorithms"],
        "category": "MEDIUM",
        "domain": "quantum"
    },
]


# =============================================================================
# LOW RESONANCE CASES (20)
# Query and context are unrelated
# Expected: LOW R, LOW P_born
# =============================================================================

LOW_RESONANCE_CASES: List[Dict[str, Any]] = [
    {
        "id": "L01",
        "query": "quantum entanglement",
        "context": ["cooking recipes", "sports statistics", "music theory"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L02",
        "query": "machine learning",
        "context": ["ancient history", "botanical gardens", "fashion design"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L03",
        "query": "database optimization",
        "context": ["poetry writing", "wildlife photography", "yoga meditation"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L04",
        "query": "network security",
        "context": ["baking bread", "gardening tips", "pet care"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L05",
        "query": "blockchain technology",
        "context": ["knitting patterns", "bird watching", "folk dancing"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L06",
        "query": "surgical procedure",
        "context": ["video games", "skateboarding", "comic books"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L07",
        "query": "legal contract",
        "context": ["ocean surfing", "mountain climbing", "sky diving"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L08",
        "query": "automotive engineering",
        "context": ["classical music", "abstract painting", "dance choreography"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L09",
        "query": "pharmaceutical research",
        "context": ["comic strips", "board games", "magic tricks"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L10",
        "query": "aerospace design",
        "context": ["cake decorating", "flower arranging", "candle making"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L11",
        "query": "programming language",
        "context": ["wine tasting", "cheese making", "perfume creation"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L12",
        "query": "statistical analysis",
        "context": ["interior decoration", "fashion modeling", "hair styling"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L13",
        "query": "mathematical proof",
        "context": ["scuba diving", "horse riding", "archery practice"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L14",
        "query": "genetic engineering",
        "context": ["antique collecting", "stamp collecting", "coin collecting"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L15",
        "query": "climate science",
        "context": ["puppet theater", "balloon animals", "face painting"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L16",
        "query": "economic theory",
        "context": ["origami folding", "calligraphy art", "pottery making"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L17",
        "query": "neural networks",
        "context": ["wedding planning", "birthday parties", "holiday decorations"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L18",
        "query": "compiler optimization",
        "context": ["tea ceremony", "meditation retreat", "spa treatments"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L19",
        "query": "cryptographic protocol",
        "context": ["fairy tales", "nursery rhymes", "bedtime stories"],
        "category": "LOW",
        "domain": "mixed"
    },
    {
        "id": "L20",
        "query": "satellite communication",
        "context": ["picnic planning", "barbecue recipes", "camping tips"],
        "category": "LOW",
        "domain": "mixed"
    },
]


# =============================================================================
# EDGE/ADVERSARIAL CASES (10)
# Tricky cases where intuition might fail
# Expected: Uncertain - interesting for analysis
# =============================================================================

EDGE_CASES: List[Dict[str, Any]] = [
    # Semantic opposites (topically similar but meaning-opposed)
    {
        "id": "E01",
        "query": "black",
        "context": ["white"],
        "category": "EDGE",
        "note": "Semantic opposites - high topic, opposite meaning"
    },
    {
        "id": "E02",
        "query": "good",
        "context": ["bad", "evil", "wrong"],
        "category": "EDGE",
        "note": "Antonyms - same domain, opposite values"
    },
    {
        "id": "E03",
        "query": "love",
        "context": ["hate", "disgust", "aversion"],
        "category": "EDGE",
        "note": "Emotional opposites"
    },
    # False statements (high topic overlap, semantically false)
    {
        "id": "E04",
        "query": "the sky is green",
        "context": ["sky color", "blue sky", "atmospheric optics"],
        "category": "EDGE",
        "note": "False claim about topic"
    },
    {
        "id": "E05",
        "query": "fish can fly",
        "context": ["fish biology", "aquatic life", "underwater creatures"],
        "category": "EDGE",
        "note": "False biological claim"
    },
    # Negations
    {
        "id": "E06",
        "query": "not a cat",
        "context": ["cat", "feline", "kitten"],
        "category": "EDGE",
        "note": "Negation of topic"
    },
    {
        "id": "E07",
        "query": "impossible task",
        "context": ["possible solutions", "achievable goals", "realistic targets"],
        "category": "EDGE",
        "note": "Negation vs affirmation"
    },
    # Homonyms/polysemy
    {
        "id": "E08",
        "query": "bank",
        "context": ["river bank", "riverbed", "waterside"],
        "category": "EDGE",
        "note": "Polysemy - financial vs geographical"
    },
    {
        "id": "E09",
        "query": "apple",
        "context": ["iPhone", "MacBook", "Steve Jobs"],
        "category": "EDGE",
        "note": "Polysemy - fruit vs company"
    },
    # Self-reference
    {
        "id": "E10",
        "query": "this sentence is false",
        "context": ["true statements", "logical validity", "semantic truth"],
        "category": "EDGE",
        "note": "Paradox/self-reference"
    },
]


# =============================================================================
# Combined test cases
# =============================================================================

def get_all_test_cases() -> List[Dict[str, Any]]:
    """Return all 100 test cases."""
    return (
        HIGH_RESONANCE_CASES +
        MEDIUM_RESONANCE_CASES +
        LOW_RESONANCE_CASES +
        EDGE_CASES
    )


def get_test_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """Return test cases for a specific category."""
    category = category.upper()
    if category == "HIGH":
        return HIGH_RESONANCE_CASES
    elif category == "MEDIUM":
        return MEDIUM_RESONANCE_CASES
    elif category == "LOW":
        return LOW_RESONANCE_CASES
    elif category == "EDGE":
        return EDGE_CASES
    else:
        return []


if __name__ == "__main__":
    print("Q44 Test Cases Summary")
    print("=" * 50)
    print(f"HIGH resonance cases: {len(HIGH_RESONANCE_CASES)}")
    print(f"MEDIUM resonance cases: {len(MEDIUM_RESONANCE_CASES)}")
    print(f"LOW resonance cases: {len(LOW_RESONANCE_CASES)}")
    print(f"EDGE/adversarial cases: {len(EDGE_CASES)}")
    print(f"TOTAL: {len(get_all_test_cases())}")

    # Sample output
    print("\nSample cases:")
    for case in get_all_test_cases()[:3]:
        print(f"  {case['id']}: {case['query'][:30]}... -> {case['category']}")
