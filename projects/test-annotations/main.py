#!/usr/bin/env python3
"""
CodeQual Test Annotations - Survey Generation & Conversion CLI

Phase 2 of CodeQual Dataset Enhancement: Creates Qualtrics surveys for human annotation
of code quality test sets with 5-dimensional assessment and converts completed surveys back to JSONL.

Usage Examples:
  # Test Qualtrics configuration
  python main.py test-config
  
  # Create survey from single source
  python main.py create-survey --source codeeval --title "CodeEval Code Quality Assessment"
  
  # Create survey from multiple sources
  python main.py create-survey --source codeeval --source codenet --title "Multi-source Code Quality Study"
  
  # Create survey with preview (first 5 samples only)
  python main.py create-survey --source codeeval --title "Test Survey" --preview
  
  # Convert completed Qualtrics survey to JSONL (using survey ID for automatic mapping)
  python main.py convert-survey --input survey_results.csv --survey-id survey_20240813_001 --output human_scores.jsonl
  
  # Validate test set compatibility
  python main.py validate-testset --source codeeval
  
  # Use custom .env file
  python main.py test-config --env-file /path/to/custom/.env
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Project root directory (where main.py is located)
ROOT = Path(__file__).parent.absolute()

# Set global ROOT in src module
import src
src.ROOT = ROOT

from src.processors.qualtrics_converter import QualtricsConverter
from src.qualtrics.survey_builder import create_survey_from_config
from src.config.qualtrics_config import load_qualtrics_config


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def generate_survey_id() -> str:
    """Generate a unique survey ID with timestamp and UUID."""
    timestamp = datetime.now().strftime("%Y%m%d")
    short_uuid = str(uuid.uuid4())[:8]
    return f"survey_{timestamp}_{short_uuid}"


def get_metadata_file() -> Path:
    """Get path to survey metadata file."""
    generated_dir = ROOT / "generated"
    generated_dir.mkdir(exist_ok=True)
    return generated_dir / "survey_metadata.jsonl"


def save_survey_metadata(survey_id: str, metadata: Dict[str, Any]):
    """Save survey metadata to JSONL file."""
    metadata_file = get_metadata_file()
    
    record = {
        "survey_id": survey_id,
        "created_at": datetime.now().isoformat(),
        **metadata
    }
    
    with open(metadata_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_survey_metadata(survey_id: str) -> Dict[str, Any]:
    """Load metadata for a specific survey ID."""
    metadata_file = get_metadata_file()
    
    if not metadata_file.exists():
        raise ValueError(f"Survey metadata file not found: {metadata_file}")
    
    with open(metadata_file, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            if record["survey_id"] == survey_id:
                return record
    
    raise ValueError(f"Survey ID not found: {survey_id}")


def load_survey_config(survey_id: str) -> Dict[str, Any]:
    """Load survey configuration for a specific survey ID."""
    config_file = ROOT / "generated/surveys" / f"{survey_id}.json"
    
    if not config_file.exists():
        raise ValueError(f"Survey configuration not found: {config_file}")
    
    with open(config_file, "r") as f:
        return json.load(f)


def validate_sources(sources: List[str]) -> bool:
    """Validate that all specified sources have test.jsonl files."""
    logger = logging.getLogger(__name__)
    base_path = ROOT.parent / "dataset/generated/integrated"
    
    for source in sources:
        test_file = base_path / source / "test.jsonl"
        if not test_file.exists():
            logger.error(f"Test file not found for source '{source}': {test_file}")
            return False
        
        # Check if file has content
        try:
            with open(test_file, "r") as f:
                first_line = f.readline()
                if not first_line.strip():
                    logger.error(f"Test file is empty for source '{source}': {test_file}")
                    return False
        except Exception as e:
            logger.error(f"Error reading test file for source '{source}': {e}")
            return False
    
    return True


def load_test_samples(sources: List[str]) -> List[Dict[str, Any]]:
    """Load test samples from specified sources."""
    logger = logging.getLogger(__name__)
    base_path = ROOT.parent / "dataset/generated/integrated"
    
    all_samples = []
    
    for source in sources:
        test_file = base_path / source / "test.jsonl"
        logger.info(f"Loading samples from {source}: {test_file}")
        
        samples_count = 0
        with open(test_file, "r") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line.strip())
                    sample["source"] = source  # Add source information
                    all_samples.append(sample)
                    samples_count += 1
        
        logger.info(f"Loaded {samples_count} samples from {source}")
    
    return all_samples


def create_survey_workflow(args, qualtrics_config=None):
    """Create a new Qualtrics survey from test samples."""
    logger = logging.getLogger(__name__)
    
    # Validate sources
    if not validate_sources(args.source):
        logger.error("Source validation failed")
        return 1
    
    # Generate survey ID
    survey_id = generate_survey_id()
    logger.info(f"Generated survey ID: {survey_id}")
    
    # Load test samples
    logger.info(f"Loading test samples from sources: {args.source}")
    try:
        all_test_samples = load_test_samples(args.source)
        logger.info(f"Loaded {len(all_test_samples)} total samples")
    except Exception as e:
        logger.error(f"Failed to load test samples: {e}")
        return 1
    
    # Select preview samples if in preview mode
    if args.preview:
        selected_samples = []
        samples_per_source = {}
        
        for source in args.source:
            source_samples = [s for s in all_test_samples if s["source"] == source]
            preview_count = min(5, len(source_samples))
            selected_samples.extend(source_samples[:preview_count])
            samples_per_source[source] = preview_count
        
        logger.info(f"Preview mode: Selected {len(selected_samples)} samples")
        for source, count in samples_per_source.items():
            logger.info(f"  {source}: {count} samples")
        
        test_samples = selected_samples
    else:
        test_samples = all_test_samples
    
    # Prepare survey metadata
    metadata = {
        "title": args.title,
        "sources": args.source,
        "total_samples": len(test_samples),
        "samples_per_source": {
            source: len([s for s in test_samples if s["source"] == source])
            for source in args.source
        },
        "qualtrics_survey_id": None,  # Will be filled when actually created
        "status": "configured"
    }
    
    # Save metadata
    save_survey_metadata(survey_id, metadata)
    logger.info(f"Survey metadata saved to: {get_metadata_file()}")
    
    # Create survey configuration file
    survey_config_dir = ROOT / "generated/surveys"
    survey_config_dir.mkdir(parents=True, exist_ok=True)
    
    survey_config = {
        "survey_id": survey_id,
        "metadata": metadata,
        "samples": test_samples,
        "survey_structure": {
            "introduction": {
                "title": f"Code Quality Assessment: {args.title}",
                "instructions": [
                    "You will evaluate code samples across 5 quality dimensions:",
                    "1. Functionality - Does the code work correctly?",
                    "2. Readability - How clear and maintainable is the code?", 
                    "3. Idiomatic - Does it follow language best practices?",
                    "4. Error Handling - How well does it manage edge cases?",
                    "5. Efficiency - Is it scalable and well-performing?",
                    "",
                    "Rate each dimension on a scale of 1-5 where:",
                    "1 = Very Poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent"
                ]
            },
            "questions": [
                {
                    "sample_id": f"{sample['problem_id']}_{sample['submission_id']}",
                    "problem_description": sample.get("problem", ""),
                    "code": sample.get("submission", ""),
                    "source": sample["source"],
                    "metadata": sample.get("metadata", {}),
                    "dimensions": [
                        {"name": "functionality", "scale": "1-5"},
                        {"name": "readability", "scale": "1-5"},
                        {"name": "idiomatic", "scale": "1-5"},
                        {"name": "error_handling", "scale": "1-5"},
                        {"name": "efficiency", "scale": "1-5"}
                    ]
                }
                for sample in test_samples
            ]
        }
    }
    
    # Save survey configuration
    config_file = survey_config_dir / f"{survey_id}.json"
    with open(config_file, "w") as f:
        json.dump(survey_config, f, indent=2)
    
    logger.info(f"Survey configuration saved: {config_file}")
    
    if args.preview:
        logger.info(f"Preview mode: Survey contains {len(test_samples)} samples (first 5 from each source)")
        logger.info(f"Full survey would contain {len(all_test_samples)} samples")
    
    # Create actual Qualtrics survey
    logger.info("Creating survey in Qualtrics...")
    try:
        qualtrics_result = create_survey_from_config(qualtrics_config, survey_config)
        
        # Update metadata with Qualtrics survey ID
        updated_metadata = metadata.copy()
        updated_metadata.update({
            "qualtrics_survey_id": qualtrics_result["survey_id"],
            "survey_link": qualtrics_result["survey_link"],
            "status": "active"
        })
        
        # Update the metadata file
        save_survey_metadata(survey_id, updated_metadata)
        
        # Update the survey config file with Qualtrics info
        survey_config["qualtrics_info"] = qualtrics_result
        with open(config_file, "w") as f:
            json.dump(survey_config, f, indent=2)
        
        logger.info("Qualtrics survey created successfully!")
        
        # Output survey information
        if args.preview:
            print(f"\n✅ Preview survey created successfully in Qualtrics!")
            print(f"Survey ID: {survey_id}")
            print(f"Qualtrics Survey ID: {qualtrics_result['survey_id']}")
            print(f"Sources: {', '.join(args.source)}")
            print(f"Total samples: {len(test_samples)} (preview: first 5 from each source)")
            print(f"Survey Link: {qualtrics_result['survey_link']}")
            print(f"Note: This is a preview survey with the first 5 samples from each source for testing.")
        else:
            print(f"\n✅ Survey created successfully in Qualtrics!")
            print(f"Survey ID: {survey_id}")
            print(f"Qualtrics Survey ID: {qualtrics_result['survey_id']}")
            print(f"Sources: {', '.join(args.source)}")
            print(f"Total samples: {len(test_samples)}")
            print(f"Survey Link: {qualtrics_result['survey_link']}")
            print(f"To convert completed survey: python main.py convert-survey --input results.csv --survey-id {survey_id}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to create Qualtrics survey: {e}")
        logger.info("Survey configuration was saved locally, but Qualtrics creation failed")
        
        print(f"\n⚠️  Survey configuration created, but Qualtrics creation failed:")
        print(f"Error: {e}")
        print(f"Survey ID: {survey_id}")
        print(f"Sources: {', '.join(args.source)}")
        print(f"Total samples: {len(test_samples)}")
        print(f"You can retry creating the survey later or use the configuration manually.")
        
        return 1




def convert_survey_workflow(args):
    """Convert completed Qualtrics survey CSV to CodeQual JSONL format."""
    logger = logging.getLogger(__name__)
    
    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    logger.info(f"Converting Qualtrics survey: {args.input}")
    logger.info(f"Using survey ID for mapping: {args.survey_id}")
    logger.info(f"Output file: {args.output}")
    
    try:
        # Load survey configuration for mapping
        survey_config = load_survey_config(args.survey_id)
        logger.info(f"Loaded survey configuration: {args.survey_id}")
        
        # Run conversion
        converter = QualtricsConverter(args.input, survey_config)
        assessments = converter.convert()
        
        if not assessments:
            logger.error("No assessments were successfully converted")
            return 1
        
        # Save results
        converter.save_detailed_jsonl(args.output)
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"📊 {len(assessments)} human assessments converted")
        print(f"📁 Output file: {args.output}")
        print(f"🔗 Survey ID: {args.survey_id}")
        
        return 0
        
    except ValueError as e:
        logger.error(f"Survey configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1


def validate_testset_workflow(args):
    """Validate test set compatibility for survey generation."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Validating test set for source: {args.source}")
    
    # Check if source test file exists and has required structure
    base_path = ROOT.parent / "dataset/generated/integrated"
    test_file = base_path / args.source / "test.jsonl"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return 1
    
    # Validate sample structure
    required_fields = ["problem_id", "submission_id", "problem", "submission"]
    sample_count = 0
    issues = []
    
    try:
        with open(test_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line.strip())
                    sample_count += 1
                    
                    # Check required fields
                    missing_fields = [field for field in required_fields if field not in sample]
                    if missing_fields:
                        issues.append(f"Line {line_num}: Missing fields: {missing_fields}")
                    
                    # Check if code and problem are non-empty
                    if not sample.get("submission", "").strip():
                        issues.append(f"Line {line_num}: Empty submission code")
                    
                    if not sample.get("problem", "").strip():
                        issues.append(f"Line {line_num}: Empty problem description")
                        
                except json.JSONDecodeError as e:
                    issues.append(f"Line {line_num}: Invalid JSON: {e}")
                
                # Only check first 100 samples for efficiency
                if sample_count >= 100 and not issues:
                    break
        
        # Report results
        print(f"Test Set Validation: {args.source}")
        print("=" * 40)
        print(f"File: {test_file}")
        print(f"Total samples: {sample_count}")
        
        if issues:
            print(f"Issues found: {len(issues)}")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
            return 1
        else:
            print("✅ Test set is valid for survey generation")
            print(f"All {sample_count} samples have required structure")
            return 0
            
    except Exception as e:
        logger.error(f"Error validating test set: {e}")
        return 1


def list_surveys_workflow(args, qualtrics_config):
    """List surveys in Qualtrics account."""
    logger = logging.getLogger(__name__)
    
    try:
        import requests
        
        headers = qualtrics_config.get_api_headers()
        url = f"{qualtrics_config.get_api_base_url()}/surveys"
        
        print("Fetching surveys from Qualtrics...")
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            surveys = data.get('result', {}).get('elements', [])
            
            print(f"\n📋 Surveys in Qualtrics Account ({len(surveys)} total)")
            print("=" * 80)
            
            if not surveys:
                print("No surveys found in your Qualtrics account.")
                return 0
            
            # Sort by creation date (newest first)
            surveys.sort(key=lambda x: x.get('creationDate', ''), reverse=True)
            
            for survey in surveys:
                survey_id = survey.get('id', 'Unknown')
                name = survey.get('name', 'Unnamed Survey')
                creation_date = survey.get('creationDate', 'Unknown')
                is_active = survey.get('isActive', False)
                status = "🟢 Active" if is_active else "🔴 Inactive"
                
                print(f"Survey ID: {survey_id}")
                print(f"  Name: {name}")
                print(f"  Created: {creation_date}")
                print(f"  Status: {status}")
                print()
            
            return 0
        else:
            print(f"❌ Failed to fetch surveys!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            return 1
            
    except ImportError:
        print("❌ requests library not available")
        print("   Install requests to use this command: pip install requests")
        return 1
    except Exception as e:
        print(f"❌ Failed to list surveys: {e}")
        return 1


def test_config_workflow(args, qualtrics_config):
    """Test Qualtrics API configuration and connectivity."""
    logger = logging.getLogger(__name__)
    
    print("Qualtrics Configuration Test")
    print("=" * 40)
    print(f"Configuration: {qualtrics_config}")
    print(f"API Base URL: {qualtrics_config.get_api_base_url()}")
    print()
    
    # Test basic API connectivity (if requests is available)
    try:
        import requests
        
        # Test endpoint - get user info
        headers = qualtrics_config.get_api_headers()
        url = f"{qualtrics_config.get_api_base_url()}/whoami"
        
        print("Testing API connectivity...")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API connection successful!")
            print(f"   User ID: {data.get('result', {}).get('userId', 'Unknown')}")
            print(f"   Brand ID: {data.get('result', {}).get('brandId', 'Unknown')}")
            return 0
        else:
            print(f"❌ API connection failed!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            return 1
            
    except ImportError:
        print("⚠️  requests library not available - skipping API connectivity test")
        print("   Install requests to test API connectivity: pip install requests")
        return 0
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="CodeQual Test Annotations - Qualtrics Survey Generation & Conversion"
    )
    
    # Global arguments
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO',
                       help='Logging level (default: INFO)')
    parser.add_argument('--env-file',
                       help='Path to .env file (default: .env in project root)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create survey command
    create_parser = subparsers.add_parser(
        'create-survey',
        help='Create a new Qualtrics survey from test samples'
    )
    create_parser.add_argument('--source', action='append', required=True,
                              help='Source dataset name (can be specified multiple times)')
    create_parser.add_argument('--title', required=True,
                              help='Survey title for human annotators')
    create_parser.add_argument('--preview', action='store_true',
                              help='Create preview with first 5 samples only')
    create_parser.set_defaults(workflow='create-survey')
    
    
    # Convert survey command
    convert_parser = subparsers.add_parser(
        'convert-survey',
        help='Convert completed Qualtrics survey CSV to CodeQual JSONL'
    )
    convert_parser.add_argument('--input', required=True,
                               help='Path to Qualtrics CSV export file')
    convert_parser.add_argument('--survey-id', required=True,
                               help='Survey ID for automatic sample mapping')
    convert_parser.add_argument('--output', default='human_scores_detailed.jsonl',
                               help='Output JSONL file path (default: human_scores_detailed.jsonl)')
    convert_parser.set_defaults(workflow='convert-survey')
    
    # Validate test set command
    validate_parser = subparsers.add_parser(
        'validate-testset',
        help='Validate test set compatibility for survey generation'
    )
    validate_parser.add_argument('--source', required=True,
                                help='Source dataset name to validate')
    validate_parser.set_defaults(workflow='validate-testset')
    
    # List surveys command
    list_surveys_parser = subparsers.add_parser(
        'list-surveys',
        help='List surveys in Qualtrics account'
    )
    list_surveys_parser.set_defaults(workflow='list-surveys')
    
    # Test configuration command
    test_config_parser = subparsers.add_parser(
        'test-config',
        help='Test Qualtrics API configuration and connectivity'
    )
    test_config_parser.set_defaults(workflow='test-config')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load Qualtrics configuration (for commands that need it)
    qualtrics_config = None
    if args.command in ['create-survey', 'test-config', 'list-surveys']:  # Commands that need Qualtrics API access
        try:
            qualtrics_config = load_qualtrics_config(args.env_file)
            logger.info("Qualtrics configuration loaded successfully")
        except ValueError as e:
            logger.error(f"Qualtrics configuration error: {e}")
            return 1
    
    try:
        # Route to appropriate workflow
        if args.command == 'create-survey':
            return create_survey_workflow(args, qualtrics_config)
        elif args.command == 'convert-survey':
            return convert_survey_workflow(args)
        elif args.command == 'validate-testset':
            return validate_testset_workflow(args)
        elif args.command == 'test-config':
            return test_config_workflow(args, qualtrics_config)
        elif args.command == 'list-surveys':
            return list_surveys_workflow(args, qualtrics_config)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())