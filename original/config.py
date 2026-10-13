PROPERTY_FULL_NAMES = {
    "plogp": ["Penalized octanol-water partition coefficient (penalized logP)", "Penalized logP", 
              "Penalized logP which is logP penalized by synthetic accessibility score and number of large rings"],
    "qed": ["QED", "Quantitative Estimate of Drug-likeness (QED)", "drug-likeness quantified by QED score"],
    "drd2": ["DRD2 inhibition", "Dopamine receptor D2 inhibition probability", "inhibition probability of Dopamine receptor D2"],
    "bbbp": ["BBB permeability", "BBBP", "Blood-brain barrier permeability (BBBP)"],
    "mutagenicity": ["Mutagenicity", "Mutagenicity predicted by Ames test", "probability to induce genetic alterations (mutagenicity)"],
    "hia": ["Intestinal adsorption", "probability to be absorbed in the intestine", "human intestinal adsorption ability"],
}

TARGET_TASKS = ["bbbp+drd2+plogp", "bbbp+drd2+qed", "bbbp+plogp+qed", "drd2+plogp+qed", "bbbp+drd2+plogp+qed",
                "mutagenicity+plogp+qed", 
                "bbbp+drd2+mutagenicity+qed",
                "bbbp+hia+mutagenicity+qed",
                "bbbp+mutagenicity+plogp+qed",
                "hia+mutagenicity+plogp+qed",
                ]

# Define property improvement thresholds except mutagenicity which should be minimized
PROPERTY_IMPV_THRESHOLDS = {
    "bbbp": 0.2,
    "drd2": 0.2,
    "hia": 0.1,
    "mutagenicity": 0.1,
    "plogp": 1.0,
    "qed": 0.1,
}
# Define property test thresholds to include a molecule in the test set
DEFAULT_PROPERTY_TEST_THRESHOLDS = {
    "bbbp": 0.5,
    "drd2": 0.1,
    "hia": 0.6,
    "mutagenicity": 0.5,
    "plogp": 0,
    "qed": 0.6,
}

# Task-specific property test thresholds
PROPERTY_TEST_THRESHOLDS = {
    "bbbp+drd2+qed": {
        "bbbp": 0.6,
        "drd2": 0.1,
        "qed": 0.4,
    },
    "bbbp+plogp+qed": {
        "bbbp": 0.5,
        "plogp": -1.5,
        "qed": 0.7,
    },
    "drd2+plogp+qed": {
        "drd2": 0.1,
        "plogp": -0.8,
        "qed": 0.5,
    },
    "bbbp+drd2+plogp": {
        "bbbp": 0.5,
        "drd2": 0.1,
        "plogp": -0.2,
    },
    "bbbp+drd2+plogp+qed": {
        "bbbp": 0.5,
        "drd2": 0.1,
        "plogp": -1.0,
        "qed": 0.4,
    },
    "mutagenicity+plogp+qed": {
        "mutagenicity": 0.5,
        "plogp": -0.5,
        "qed": 0.7,
    },
    "hia+mutagenicity+plogp+qed": {
        "hia": 0.7,
        "mutagenicity": 0.5,
        "plogp": -2.0,
        "qed": 0.6,
    },
    "bbbp+mutagenicity+plogp+qed": {
        "bbbp": 0.5,
        "mutagenicity": 0.5,
        "plogp": -0.7,
        "qed": 0.7,
    },
    "bbbp+drd2+mutagenicity+qed": {
        "bbbp": 0.5,
        "drd2": 0.1,
        "mutagenicity": 0.4,
        "qed": 0.3,
    },
    "bbbp+hia+mutagenicity+qed": {
        "bbbp": 0.4,
        "hia": 0.7,
        "mutagenicity": 0.4,
        "qed": 0.7,
    },
}

DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_MAX_TOKENS = 1024