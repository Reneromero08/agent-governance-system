#!/usr/bin/env python3
"""Generate Large Anchor Sets for Cross-Model Alignment Research.

Creates anchor sets of various sizes (500, 1000, 2000) using
common English words across diverse semantic categories.

Usage:
    python large_anchor_generator.py
"""

import hashlib
from typing import List

# =============================================================================
# EXPANDED ANCHOR SETS
# =============================================================================

# Core categories (8 words each)
CONCRETE_NOUNS = [
    "dog", "cat", "tree", "house", "car", "book", "water", "food",
    "bird", "fish", "flower", "mountain", "river", "ocean", "forest", "garden",
    "chair", "table", "bed", "door", "window", "wall", "floor", "roof",
    "sun", "moon", "star", "cloud", "rain", "snow", "wind", "fire",
]

ABSTRACT_CONCEPTS = [
    "love", "hate", "fear", "joy", "time", "space", "truth", "idea",
    "hope", "faith", "peace", "war", "freedom", "justice", "power", "knowledge",
    "beauty", "wisdom", "courage", "honor", "pride", "shame", "guilt", "anger",
    "happiness", "sadness", "excitement", "boredom", "curiosity", "confusion", "surprise", "disgust",
]

ACTIONS = [
    "run", "walk", "jump", "climb", "swim", "fly", "crawl", "dance",
    "think", "speak", "write", "read", "listen", "watch", "learn", "teach",
    "create", "destroy", "build", "break", "fix", "change", "grow", "shrink",
    "give", "take", "share", "keep", "find", "lose", "hide", "show",
    "eat", "drink", "sleep", "wake", "work", "play", "rest", "move",
    "push", "pull", "lift", "drop", "throw", "catch", "hold", "release",
]

PROPERTIES = [
    "big", "small", "tall", "short", "long", "wide", "narrow", "deep",
    "fast", "slow", "quick", "steady", "smooth", "rough", "soft", "hard",
    "hot", "cold", "warm", "cool", "wet", "dry", "fresh", "stale",
    "bright", "dark", "light", "heavy", "loud", "quiet", "sharp", "dull",
    "good", "bad", "right", "wrong", "true", "false", "real", "fake",
    "old", "young", "new", "ancient", "modern", "classic", "simple", "complex",
]

RELATIONS = [
    "above", "below", "over", "under", "beside", "between", "among", "around",
    "inside", "outside", "within", "beyond", "through", "across", "along", "against",
    "before", "after", "during", "while", "since", "until", "when", "where",
    "with", "without", "for", "against", "toward", "away", "into", "onto",
    "near", "far", "close", "distant", "left", "right", "front", "back",
]

NUMBERS_QUANTITIES = [
    "one", "two", "three", "four", "five", "ten", "hundred", "thousand",
    "all", "none", "some", "many", "few", "several", "most", "least",
    "more", "less", "equal", "different", "same", "similar", "opposite", "other",
    "first", "last", "next", "previous", "beginning", "end", "middle", "edge",
    "whole", "part", "half", "quarter", "double", "triple", "single", "multiple",
]

DOMAINS = [
    "science", "art", "music", "math", "language", "history", "geography", "philosophy",
    "physics", "chemistry", "biology", "astronomy", "psychology", "sociology", "economics", "politics",
    "medicine", "engineering", "law", "business", "education", "religion", "technology", "nature",
    "literature", "poetry", "drama", "comedy", "tragedy", "fiction", "reality", "fantasy",
    "sport", "game", "dance", "song", "story", "picture", "sculpture", "architecture",
]

PEOPLE_ROLES = [
    "mother", "father", "child", "parent", "sibling", "friend", "enemy", "stranger",
    "teacher", "student", "doctor", "patient", "leader", "follower", "worker", "manager",
    "artist", "scientist", "writer", "reader", "speaker", "listener", "buyer", "seller",
    "king", "queen", "prince", "princess", "knight", "soldier", "farmer", "merchant",
    "hero", "villain", "victim", "witness", "judge", "lawyer", "criminal", "police",
]

BODY_PARTS = [
    "head", "face", "eye", "ear", "nose", "mouth", "hand", "foot",
    "arm", "leg", "finger", "toe", "heart", "brain", "bone", "skin",
    "hair", "teeth", "tongue", "throat", "chest", "back", "shoulder", "knee",
]

ANIMALS = [
    "lion", "tiger", "bear", "wolf", "fox", "deer", "horse", "cow",
    "pig", "sheep", "goat", "chicken", "duck", "eagle", "hawk", "owl",
    "snake", "frog", "turtle", "whale", "dolphin", "shark", "crab", "spider",
    "bee", "butterfly", "ant", "mouse", "rabbit", "squirrel", "monkey", "elephant",
]

PLANTS = [
    "rose", "lily", "daisy", "tulip", "orchid", "sunflower", "violet", "jasmine",
    "oak", "pine", "maple", "palm", "bamboo", "willow", "birch", "cedar",
    "grass", "moss", "fern", "vine", "bush", "shrub", "weed", "herb",
    "apple", "orange", "banana", "grape", "lemon", "cherry", "peach", "plum",
]

FOOD_DRINK = [
    "bread", "rice", "meat", "fish", "cheese", "egg", "milk", "butter",
    "salt", "sugar", "pepper", "oil", "vinegar", "honey", "cream", "flour",
    "coffee", "tea", "juice", "wine", "beer", "soup", "sauce", "salad",
    "cake", "cookie", "candy", "chocolate", "ice", "fruit", "vegetable", "grain",
]

MATERIALS = [
    "wood", "stone", "metal", "glass", "paper", "cloth", "leather", "plastic",
    "gold", "silver", "iron", "copper", "steel", "bronze", "aluminum", "tin",
    "brick", "concrete", "cement", "sand", "clay", "mud", "dirt", "dust",
    "rubber", "cotton", "wool", "silk", "nylon", "foam", "wax", "oil",
]

PLACES = [
    "home", "school", "office", "hospital", "church", "temple", "park", "beach",
    "city", "town", "village", "country", "state", "nation", "continent", "world",
    "street", "road", "bridge", "tunnel", "station", "airport", "port", "market",
    "farm", "factory", "mine", "warehouse", "store", "restaurant", "hotel", "museum",
]

WEATHER_NATURE = [
    "storm", "thunder", "lightning", "earthquake", "volcano", "flood", "drought", "hurricane",
    "sunrise", "sunset", "dawn", "dusk", "noon", "midnight", "morning", "evening",
    "spring", "summer", "autumn", "winter", "season", "climate", "weather", "temperature",
    "ice", "frost", "fog", "mist", "dew", "hail", "sleet", "breeze",
]

TOOLS_OBJECTS = [
    "knife", "fork", "spoon", "plate", "cup", "bowl", "pot", "pan",
    "hammer", "nail", "screw", "drill", "saw", "axe", "shovel", "rake",
    "pen", "pencil", "brush", "ruler", "scissors", "tape", "glue", "string",
    "key", "lock", "chain", "rope", "wire", "pipe", "tube", "box",
    "clock", "watch", "mirror", "lamp", "phone", "camera", "radio", "screen",
]

CLOTHING = [
    "shirt", "pants", "dress", "skirt", "coat", "jacket", "sweater", "vest",
    "hat", "cap", "scarf", "glove", "sock", "shoe", "boot", "sandal",
    "belt", "tie", "button", "zipper", "pocket", "collar", "sleeve", "hem",
]

COLORS_PATTERNS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "silver", "gold", "bronze", "copper", "cream",
    "stripe", "dot", "check", "plain", "pattern", "solid", "mixed", "faded",
]

DIRECTIONS_POSITIONS = [
    "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
    "up", "down", "forward", "backward", "sideways", "diagonal", "vertical", "horizontal",
    "top", "bottom", "center", "corner", "side", "surface", "interior", "exterior",
]

TIME_WORDS = [
    "second", "minute", "hour", "day", "week", "month", "year", "decade",
    "century", "millennium", "moment", "instant", "period", "era", "age", "epoch",
    "past", "present", "future", "today", "tomorrow", "yesterday", "now", "then",
    "always", "never", "sometimes", "often", "rarely", "usually", "frequently", "occasionally",
]

MENTAL_STATES = [
    "thought", "idea", "memory", "dream", "imagination", "reason", "logic", "intuition",
    "belief", "doubt", "certainty", "confusion", "understanding", "knowledge", "ignorance", "wisdom",
    "attention", "focus", "concentration", "distraction", "awareness", "consciousness", "unconscious", "subconscious",
]

SOCIAL_WORDS = [
    "family", "friend", "neighbor", "community", "society", "culture", "tradition", "custom",
    "law", "rule", "right", "duty", "responsibility", "privilege", "permission", "prohibition",
    "agreement", "conflict", "cooperation", "competition", "negotiation", "compromise", "victory", "defeat",
]

COMMUNICATION = [
    "word", "sentence", "paragraph", "chapter", "book", "letter", "message", "signal",
    "question", "answer", "statement", "claim", "argument", "evidence", "proof", "example",
    "meaning", "sense", "definition", "description", "explanation", "interpretation", "translation", "summary",
]

MOVEMENT_CHANGE = [
    "motion", "movement", "speed", "velocity", "acceleration", "direction", "path", "route",
    "start", "stop", "pause", "continue", "advance", "retreat", "progress", "regress",
    "increase", "decrease", "expand", "contract", "rise", "fall", "open", "close",
]

VALUE_QUALITY = [
    "value", "worth", "price", "cost", "benefit", "profit", "loss", "gain",
    "quality", "quantity", "measure", "standard", "criterion", "benchmark", "target", "goal",
    "success", "failure", "achievement", "accomplishment", "mistake", "error", "fault", "flaw",
]


def compute_anchor_hash(anchors: List[str]) -> str:
    """Compute deterministic hash of anchor set."""
    canonical = "\n".join(sorted(anchors))
    full_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return full_hash[:16]


def generate_anchor_set(size: int) -> List[str]:
    """Generate anchor set of specified size."""
    # Combine all categories
    all_words = []
    categories = [
        CONCRETE_NOUNS, ABSTRACT_CONCEPTS, ACTIONS, PROPERTIES, RELATIONS,
        NUMBERS_QUANTITIES, DOMAINS, PEOPLE_ROLES, BODY_PARTS, ANIMALS,
        PLANTS, FOOD_DRINK, MATERIALS, PLACES, WEATHER_NATURE,
        TOOLS_OBJECTS, CLOTHING, COLORS_PATTERNS, DIRECTIONS_POSITIONS,
        TIME_WORDS, MENTAL_STATES, SOCIAL_WORDS, COMMUNICATION,
        MOVEMENT_CHANGE, VALUE_QUALITY,
    ]

    for cat in categories:
        all_words.extend(cat)

    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in all_words:
        if word.lower() not in seen:
            seen.add(word.lower())
            unique_words.append(word.lower())

    if size > len(unique_words):
        print(f"Warning: Requested {size} anchors but only {len(unique_words)} unique words available")
        return unique_words

    return unique_words[:size]


# Pre-generate anchor sets
ANCHOR_128 = generate_anchor_set(128)
ANCHOR_256 = generate_anchor_set(256)
ANCHOR_512 = generate_anchor_set(512)
ANCHOR_1024 = generate_anchor_set(1024)

# Compute hashes
ANCHOR_128_HASH = compute_anchor_hash(ANCHOR_128)
ANCHOR_256_HASH = compute_anchor_hash(ANCHOR_256)
ANCHOR_512_HASH = compute_anchor_hash(ANCHOR_512)
ANCHOR_1024_HASH = compute_anchor_hash(ANCHOR_1024)


if __name__ == "__main__":
    print("=" * 60)
    print("LARGE ANCHOR SET GENERATOR")
    print("=" * 60)

    for name, anchors in [
        ("ANCHOR_128", ANCHOR_128),
        ("ANCHOR_256", ANCHOR_256),
        ("ANCHOR_512", ANCHOR_512),
        ("ANCHOR_1024", ANCHOR_1024),
    ]:
        print(f"\n{name}:")
        print(f"  Count: {len(anchors)}")
        print(f"  Hash: {compute_anchor_hash(anchors)}")
        print(f"  First 10: {anchors[:10]}")
        print(f"  Last 10: {anchors[-10:]}")

    print(f"\n\nTotal unique words available: {len(generate_anchor_set(9999))}")
