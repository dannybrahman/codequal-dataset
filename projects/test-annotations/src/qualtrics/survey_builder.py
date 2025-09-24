#!/usr/bin/env python3
"""
Qualtrics Survey Builder

Creates actual Qualtrics surveys using the API based on code quality assessment requirements.
"""

import json
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..config.qualtrics_config import QualtricsConfig


class QualtricsAPI:
    """Wrapper for Qualtrics API operations."""
    
    def __init__(self, config: QualtricsConfig):
        self.config = config
        self.base_url = config.get_api_base_url()
        self.headers = config.get_api_headers()
    
    def create_survey(self, survey_name: str, survey_description: str = None) -> str:
        """Create a new survey and return survey ID."""
        url = f"{self.base_url}/survey-definitions"
        
        payload = {
            "SurveyName": survey_name,
            "Language": "EN",
            "ProjectCategory": "CORE"
        }
        
        response = requests.post(url, headers=self.headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            survey_id = result['result']['SurveyID']
            return survey_id
        else:
            raise Exception(f"Failed to create survey: {response.status_code} - {response.text}")
    
    def create_question(self, survey_id: str, question_data: dict) -> str:
        """Create a question in the survey."""
        url = f"{self.base_url}/survey-definitions/{survey_id}/questions"
        
        response = requests.post(url, headers=self.headers, json=question_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['result']['QuestionID']
        else:
            raise Exception(f"Failed to create question: {response.status_code} - {response.text}")
    
    def update_survey_options(self, survey_id: str, options: dict):
        """Update survey options like header, footer, etc."""
        url = f"{self.base_url}/survey-definitions/{survey_id}/options"
        
        response = requests.put(url, headers=self.headers, json=options, timeout=30)
        
        if response.status_code == 200:
            return True
        else:
            raise Exception(f"Failed to update survey options: {response.status_code} - {response.text}")
    
    def get_survey(self, survey_id: str) -> dict:
        """Get survey definition."""
        url = f"{self.base_url}/survey-definitions/{survey_id}"
        response = requests.get(url, headers=self.headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get survey: {response.status_code} - {response.text}")
    
    def activate_survey(self, survey_id: str) -> bool:
        """Activate survey for data collection."""
        url = f"{self.base_url}/surveys/{survey_id}"
        payload = {"isActive": True}
        
        response = requests.put(url, headers=self.headers, json=payload, timeout=30)
        return response.status_code == 200


class SurveyBuilder:
    """Builds Qualtrics surveys for code quality assessment."""
    
    def __init__(self, config: QualtricsConfig):
        self.api = QualtricsAPI(config)
        self.config = config
    
    def create_code_quality_survey(self, survey_config: dict) -> str:
        """
        Create a complete code quality assessment survey.
        
        Args:
            survey_config: Survey configuration with metadata and samples
            
        Returns:
            Qualtrics survey ID
        """
        survey_name = survey_config['metadata']['title']
        samples = survey_config['survey_structure']['questions']
        
        # Create basic survey
        survey_id = self.api.create_survey(
            survey_name, 
            f"Code quality assessment survey with {len(samples)} samples"
        )
        
        # Update survey options with syntax highlighting and navigation
        survey_options = {
            "Header": (
                "<!-- Load Highlight.js and theme -->\n"
                "<link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css\">\n"
                "<script src=\"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js\"></script>\n"
                "<script>hljs.highlightAll();</script>"
            ),
            "Footer": "",
            "ProgressBarDisplay": "VerboseText",
            "BackButton": "true",
            "SaveAndContinue": "true",
            "SurveyProtection": "PublicSurvey",
            "QuestionsPerPage": "1",
            "PreviousButton": "true",
            "NextButton": "true",
            "SurveyTermination": "DefaultMessage",
            "SurveyExpiration": None
        }
        
        try:
            self.api.update_survey_options(survey_id, survey_options)
        except Exception as e:
            print(f"Warning: Could not update survey options: {e}")
        
        # Create instruction page as first question
        instruction_html = self._get_instruction_html()
        instruction_question = {
            "QuestionText": instruction_html,
            "QuestionType": "DB",
            "Selector": "TB",
            "DataExportTag": "Instructions",
            "Configuration": {
                "QuestionDescriptionOption": "UseText"
            },
            "Validation": {
                "Settings": {
                    "Type": "None"
                }
            }
        }
        
        try:
            instruction_id = self.api.create_question(survey_id, instruction_question)
            print(f"Created instruction page: {instruction_id}")
        except Exception as e:
            print(f"Failed to create instruction page: {e}")
        
        # Create questions for each sample
        for i, sample in enumerate(samples, 1):
            question_text = self._format_question_text(
                sample['problem_description'], 
                sample['code'],
                sample
            )
            
            question_data = {
                "QuestionText": question_text,
                "QuestionType": "Matrix",
                "Selector": "Likert",
                "SubSelector": "SingleAnswer",
                "DataExportTag": f"Q{i}",
                "Configuration": {
                    "QuestionDescriptionOption": "UseText",
                    "TextPosition": "inline",
                    "ChoiceColumnWidth": 25,
                    "RepeatHeaders": "none",
                    "WhiteSpace": "OFF",
                    "MobileFirst": True
                },
                "Choices": {
                    "1": {"Display": "Functionality"},
                    "2": {"Display": "Readability"},
                    "3": {"Display": "Idiomatic"},
                    "4": {"Display": "Efficiency"},
                    "5": {"Display": "Error-handling"}
                },
                "Answers": {
                    "1": {"Display": "1 (Poor)"},
                    "2": {"Display": "2"},
                    "3": {"Display": "3"},
                    "4": {"Display": "4"},
                    "5": {"Display": "5 (Excellent)"}
                },
                "Validation": {
                    "Settings": {
                        "ForceResponse": "ON",
                        "ForceResponseType": "ON",
                        "Type": "None"
                    }
                }
            }
            
            try:
                question_id = self.api.create_question(survey_id, question_data)
                print(f"Created question {i}/{len(samples)}: {question_id}")
            except Exception as e:
                print(f"Failed to create question {i}: {e}")
                # Continue with other questions
        
        # Activate survey
        self.api.activate_survey(survey_id)
        
        return survey_id
    
    
    def _get_instruction_html(self) -> str:
        """Get the instruction HTML for the first page of the survey."""
        return """<h2>Welcome to the Python Code Quality Evaluation Survey</h2>
<br>
<p>Thank you for participating in this study. Your input will help us better understand how well large language models are able to generate high-quality code.</p>
<br>
<p>
You will be shown a series of Python code snippets, each accompanied by a problem statement. Your task is to rate each snippet on the following five dimensions:
</p>
<ol>
  <li><strong>Functionality</strong> – Does the code work as described in the problem statement?</li>
  <li><strong>Readability</strong> – How easy is the code to understand for a human reader?</li>
  <li><strong>Idiomatic</strong> – Does the code follow idiomatic Python practices?</li>
  <li><strong>Efficiency</strong> – Is the code scalable and efficient for larger inputs or data?</li>
  <li><strong>Error handling</strong> – Does the code anticipate and handle potential edge cases or input errors?</li>
</ol>
<br>

<h3>Scoring Rubric</h3>
<p>Please use the following rubric to guide your ratings:</p>

<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
<thead>
<tr style="background-color: #f0f0f0;">
  <th style="border: 1px solid #ddd; padding: 8px; text-align: left; width: 10%;">Score</th>
  <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Functionality</th>
  <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Readability</th>
  <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Idiomatic</th>
  <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Error Handling</th>
  <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Efficiency</th>
</tr>
</thead>
<tbody>
<tr>
  <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">5</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Perfect</strong> - Solves completely and correctly with all edge cases</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Excellent</strong> - Crystal clear with meaningful names and structure</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Expert</strong> - Follows all best practices optimally</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Comprehensive</strong> - Handles all edge cases with informative errors</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Optimal</strong> - Best algorithm with excellent complexity</td>
</tr>
<tr style="background-color: #fafafa;">
  <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">4</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Good</strong> - Works correctly for main cases</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Good</strong> - Clear and well-organized</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Proficient</strong> - Generally follows conventions</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Good</strong> - Handles most error cases</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Good</strong> - Efficient with reasonable performance</td>
</tr>
<tr>
  <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">3</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Adequate</strong> - Works for typical cases with some limitations</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Adequate</strong> - Reasonably readable</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Acceptable</strong> - Follows basic conventions</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Basic</strong> - Handles some error cases</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Acceptable</strong> - Works but could be optimized</td>
</tr>
<tr style="background-color: #fafafa;">
  <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">2</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Poor</strong> - Significant bugs or only simple cases</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Poor</strong> - Hard to understand</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Poor</strong> - Ignores many conventions</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Minimal</strong> - Little error handling</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Poor</strong> - Inefficient approach</td>
</tr>
<tr>
  <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">1</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Broken</strong> - Doesn't work or has fundamental flaws</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Unreadable</strong> - Extremely difficult to understand</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Non-idiomatic</strong> - Completely ignores conventions</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>None</strong> - No error handling</td>
  <td style="border: 1px solid #ddd; padding: 8px;"><strong>Very Poor</strong> - Extremely inefficient</td>
</tr>
</tbody>
</table>

<br>
<p>Below is an example of a problem statement, a corresponding code snippet, and how it might be rated with brief justifications. This example is only meant to serve as a general guideline—your ratings do not need to follow it strictly.</p>


<div style="border: 1px solid #ccc; background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin: 20px 0;">

<p><strong>Problem:</strong> Write a program that prints multiplication tables.</p>
<p><strong>Code:</strong> </p>

<pre><code class="language-python">
M = 9
N = 9
def main():
    for i in range(1, M+1, 1):
        for j in range(1, N+1, 1):
            mult = i * j
            print(str(i) + "x" + str(j) + "=" + str(i * j))
main()
</code></pre>
<br>
<p><strong>Example Ratings with Reasoning:</strong></p>
<ol>
  <li><strong>Functionality</strong> – 5: The code correctly generates and prints the multiplication table up to M x N as requested.</li>
  <li><strong>Readability</strong> – 4: The logic is easy to follow, but the lack of comments and direct printing in loops could be slightly improved.</li>
  <li><strong>Idiomatic</strong> – 3: Uses basic Python features like `range`, but could be cleaner with f-strings instead of string concatenation.</li>
  <li><strong>Efficiency</strong> – 2: Works fine for small values of M and N, but repeated calculations in the print statement could be optimized.</li>
  <li><strong>Error handling</strong> – 3: Assumes inputs are valid. No checks for negative or non-integer values.</li>
</ol>
</div>

<br>
<p>Please use a scale of 1 (Very Poor) to 5 (Excellent) for each category. You will not be asked to provide justifications—only the ratings. The example above is just to help you understand what to look for.</p>
<br>
<p>You may complete the survey in multiple sittings. The estimated total time commitment is around 20 hours.</p>
<br>
<p><em>Click "Next" to begin the survey.</em></p>"""
    
    def _format_question_text(self, problem_description: str, code: str, sample: dict) -> str:
        """Format the question text with problem description and code."""
        # Escape HTML in problem description
        problem_html = problem_description.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Escape HTML in code but preserve structure
        code_html = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Get language from metadata, default to "Python" if not found
        language = sample.get('metadata', {}).get('language', 'python')
        # Capitalize language name for display
        language_display = language.upper() if language == 'cpp' else language.capitalize()
        if language == 'js':
            language_display = 'JavaScript'
        
        return (
            f"<h2>Problem Statement:&nbsp;</h2>\n"
            f"<p>{problem_html}</p><br>\n"
            f"<h2>{language_display} Code:</h2>\n"
            f'<pre><code class="language-{language}">\n{code_html}\n</code></pre>'
        )
    
    def get_survey_link(self, survey_id: str) -> str:
        """Get the public survey link."""
        # This would typically require another API call to get the anonymous link
        # For now, return the basic URL format
        return f"https://{self.config.datacenter}.qualtrics.com/jfe/form/{survey_id}"


def create_survey_from_config(config: QualtricsConfig, survey_config: dict) -> dict:
    """
    Create a Qualtrics survey from survey configuration.
    
    Args:
        config: Qualtrics API configuration
        survey_config: Survey configuration dict
        
    Returns:
        Dict with survey_id and survey_link
    """
    builder = SurveyBuilder(config)
    
    survey_id = builder.create_code_quality_survey(survey_config)
    survey_link = builder.get_survey_link(survey_id)
    
    return {
        "survey_id": survey_id,
        "survey_link": survey_link,
        "total_questions": len(survey_config['survey_structure']['questions'])
    }