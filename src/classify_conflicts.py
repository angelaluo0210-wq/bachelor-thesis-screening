"""Reclassify conflicting labels with strict identifier hierarchy and placeholder detection."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.prepare_dataset import load_data, normalize_text, normalize_review_key

# Placeholder title patterns
PLACEHOLDER_PATTERNS = {
    "[not available]",
    "[not available].",
    "n/a",
    "unknown",
    "not available",
    "",
    "[unknown]",
    "[no title]",
    "no title",
}

def is_placeholder_title(title: str) -> bool:
    """Check if a title is a placeholder or empty."""
    if not title or pd.isna(title):
        return True
    normalized = str(title).strip().lower()
    return normalized in PLACEHOLDER_PATTERNS or len(normalized) < 3

def get_all_identifiers(row_df: pd.DataFrame):
    """Extract all identifiers from a set of rows."""
    pmids = row_df["pmid"].fillna("").astype(str).str.strip()
    pmids = [x for x in pmids if x and x != "nan"]
    
    dois = row_df["doi"].fillna("").astype(str).str.strip()
    dois = [x for x in dois if x and x != "nan"]
    
    openalex = row_df["openalex_id"].fillna("").astype(str).str.strip()
    openalex = [x for x in openalex if x and x != "nan"]
    
    titles = row_df["title"].fillna("").astype(str).str.strip()
    titles = [x for x in titles if x and len(x) > 2]
    
    return {
        "pmids": list(set(pmids)),
        "dois": list(set(dois)),
        "openalex_ids": list(set(openalex)),
        "titles": list(set(titles)),
    }

def classify_conflicts():
    """Classify conflicts with strict identifier hierarchy."""
    
    # Load data
    df_articles, _ = load_data()
    conflicts = pd.read_csv(ROOT / "reports" / "conflicting_labels.csv")
    
    classified = []
    conflict_id = 1
    
    for idx, conflict in conflicts.iterrows():
        review_id = conflict["review_id"]
        identifier_type = conflict["identifier_type"]
        identifier_value = conflict["identifier_value"]
        title = conflict["title"]
        labels_list = eval(str(conflict["labels"])) if pd.notna(conflict["labels"]) else []
        duplicate_rows_count = conflict["duplicate_rows"]
        
        # Find all articles matching this conflict
        subset = df_articles.copy()
        
        # Step 1: Find rows matching the detected conflict identifier
        if identifier_type == "pmid":
            subset = subset[
                (subset["pmid"].fillna("").astype(str).str.strip() == str(identifier_value).strip()) |
                (subset["pmid"] == identifier_value)
            ]
            identifier_used = "pmid"
            identifier_strength = "strong"
        elif identifier_type == "doi":
            subset = subset[
                (subset["doi"].fillna("").astype(str).str.strip() == str(identifier_value).strip()) |
                (subset["doi"] == identifier_value)
            ]
            identifier_used = "doi"
            identifier_strength = "strong"
        elif identifier_type == "openalex_id":
            subset = subset[
                (subset["openalex_id"].fillna("").astype(str).str.strip() == str(identifier_value).strip()) |
                (subset["openalex_id"] == identifier_value)
            ]
            identifier_used = "openalex_id"
            identifier_strength = "strong"
        elif identifier_type == "normalized_title":
            # Title match is weak - check if it's a placeholder
            is_placeholder = is_placeholder_title(str(identifier_value))
            subset["normalized_title"] = subset["title"].fillna("").map(normalize_text)
            subset = subset[subset["normalized_title"] == identifier_value]
            identifier_used = "normalized_title"
            identifier_strength = "weak"
            
            if is_placeholder:
                identifier_strength = "placeholder"
        
        if subset.empty:
            continue
        
        # Get unique reviews with this identifier
        reviews_with_conflict = sorted(subset["review_id"].unique().tolist())
        num_reviews = len(reviews_with_conflict)
        
        # Get all identifiers in this conflict group
        identifiers = get_all_identifiers(subset)
        
        # Get labels
        labels_in_conflict = subset["label"].fillna("").astype(str)
        labels_in_conflict = [int(float(x)) for x in labels_in_conflict if x and x.strip() and x != "nan"]
        labels_in_conflict = sorted(list(set(labels_in_conflict)))
        
        # Classify based on strength and distribution
        if identifier_strength == "placeholder":
            # Placeholder title only - check if other identifiers match
            if len(identifiers["pmids"]) > 0 or len(identifiers["dois"]) > 0 or len(identifiers["openalex_ids"]) > 0:
                # Has strong identifiers despite placeholder title
                if num_reviews == 1 and len(labels_in_conflict) > 1:
                    conflict_type = "same_review_same_article_conflicting_label"
                    requires_review = True
                    notes = f"Same-review conflict; placeholder title but identified by strong identifiers"
                else:
                    conflict_type = "cross_review_same_article_different_label"
                    requires_review = False
                    notes = f"Cross-review same article (identified by strong identifiers despite placeholder title)"
            else:
                # Placeholder title with no strong identifiers - false duplicate
                conflict_type = "probable_false_duplicate_due_to_placeholder_title"
                requires_review = True
                notes = f"Multiple rows with only placeholder title match, no strong identifiers (PMID/DOI/OpenAlex)"
        
        elif identifier_strength == "weak":
            # Normalized title match - weak evidence
            if len(identifiers["pmids"]) == 0 and len(identifiers["dois"]) == 0 and len(identifiers["openalex_ids"]) == 0:
                # No strong identifiers - title-only match
                if is_placeholder_title(str(identifier_value)):
                    conflict_type = "probable_false_duplicate_due_to_placeholder_title"
                    requires_review = True
                    notes = f"Placeholder title with no identifiers across {num_reviews} review(s)"
                else:
                    # Real title but no identifiers
                    if num_reviews == 1 and len(labels_in_conflict) > 1:
                        conflict_type = "same_review_same_article_conflicting_label"
                        requires_review = True
                        notes = f"Same-review conflict identified by title normalization only (no PMID/DOI/OpenAlex)"
                    else:
                        conflict_type = "probable_false_duplicate_due_to_title_only_match"
                        requires_review = True
                        notes = f"Multiple rows with similar title but different identifiers across {num_reviews} review(s) - likely different articles"
            else:
                # Has strong identifiers in addition to title match
                if num_reviews == 1 and len(labels_in_conflict) > 1:
                    conflict_type = "same_review_same_article_conflicting_label"
                    requires_review = True
                    notes = f"Same-review conflict (identified by PMID/DOI and title normalization)"
                else:
                    conflict_type = "cross_review_same_article_different_label"
                    requires_review = False
                    notes = f"Cross-review same article ({num_reviews} reviews)"
        
        else:
            # Strong identifier (PMID, DOI, or OpenAlex ID)
            if num_reviews == 1 and len(labels_in_conflict) > 1:
                conflict_type = "same_review_same_article_conflicting_label"
                requires_review = True
                notes = f"Same-review label conflict (identified by {identifier_used})"
            elif num_reviews > 1:
                conflict_type = "cross_review_same_article_different_label"
                requires_review = False
                notes = f"Same article appears in {num_reviews} reviews with different labels"
            else:
                conflict_type = "unclear_manual_review_required"
                requires_review = True
                notes = f"Unexpected case: strong identifier but unclear status"
        
        # Compile result
        classified.append({
            "conflict_id": conflict_id,
            "review_id": review_id,
            "title": title,
            "pmid": "; ".join(identifiers["pmids"]) if identifiers["pmids"] else "",
            "doi": "; ".join(identifiers["dois"]) if identifiers["dois"] else "",
            "openalex_id": "; ".join(identifiers["openalex_ids"]) if identifiers["openalex_ids"] else "",
            "labels_found": str(sorted(labels_in_conflict)),
            "identifier_used_for_match": identifier_type,
            "identifier_strength": identifier_strength,
            "conflict_type": conflict_type,
            "requires_manual_review": requires_review,
            "notes": notes,
        })
        
        conflict_id += 1
    
    result_df = pd.DataFrame(classified)
    return result_df

if __name__ == "__main__":
    classified_df = classify_conflicts()
    output_path = ROOT / "reports" / "conflicting_labels_classified.csv"
    classified_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"CONFLICT CLASSIFICATION REPORT")
    print(f"{'='*80}\n")
    
    print(f"Total conflicts reclassified: {len(classified_df)}\n")
    
    # Count by type
    type_counts = classified_df["conflict_type"].value_counts()
    print("Classification results:")
    for conflict_type, count in type_counts.items():
        print(f"  {conflict_type}: {count}")
    
    # Count same-review conflicts
    same_review = classified_df[classified_df["conflict_type"] == "same_review_same_article_conflicting_label"]
    print(f"\n→ True same-review label conflicts: {len(same_review)}")
    
    # Count cross-review conflicts
    cross_review = classified_df[classified_df["conflict_type"] == "cross_review_same_article_different_label"]
    print(f"→ Legitimate cross-review differences: {len(cross_review)}")
    
    # Count false duplicates
    false_dup = classified_df[
        (classified_df["conflict_type"] == "probable_false_duplicate_due_to_placeholder_title") |
        (classified_df["conflict_type"] == "probable_false_duplicate_due_to_title_only_match")
    ]
    print(f"→ False duplicate detections: {len(false_dup)}")
    
    # Count requiring review
    needs_review = classified_df[classified_df["requires_manual_review"] == True]
    print(f"→ Rows requiring manual review: {len(needs_review)}")
    
    print(f"\nSaved to: {output_path}\n")
    
    print("Detailed classification:")
    print(classified_df[["conflict_id", "review_id", "conflict_type", "identifier_used_for_match", "labels_found"]].to_string(index=False))
    
    print("\n" + "="*80)
    print("FALSE DUPLICATE DETECTIONS (not same article)")
    print("="*80)
    if len(false_dup) > 0:
        print(false_dup[["conflict_id", "review_id", "title", "identifier_used_for_match", "notes"]].to_string(index=False))
    else:
        print("None")
    
    print("\n" + "="*80)
    print("SAME-REVIEW LABEL CONFLICTS (manual review required)")
    print("="*80)
    if len(same_review) > 0:
        print(same_review[["conflict_id", "review_id", "title", "identifier_strength", "notes"]].to_string(index=False))
    else:
        print("None")
    
    print("\n" + "="*80)
    print("CROSS-REVIEW SAME ARTICLE (likely valid - different criteria per review)")
    print("="*80)
    if len(cross_review) > 0:
        print(cross_review[["conflict_id", "review_id", "title", "notes"]].to_string(index=False))
    else:
        print("None")
