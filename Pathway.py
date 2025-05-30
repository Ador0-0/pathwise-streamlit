# Standard library imports
import datetime
import sqlite3
import time
import io
import json
import os
import hashlib

# Third-party imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import requests
import numpy as np

# ------------------------- Helper Functions -------------------------
# Update get_visualization_colors for more vibrant palette
def get_visualization_colors(n_colors):
    """Generate a list of user-friendly, muted colors for charts (works on both dark and light backgrounds)"""
    # Muted, accessible palette for up to 10 slices
    palette = [
        "#6b7280",  # muted slate gray
        "#60a5fa",  # soft blue
        "#a3a3a3",  # neutral gray
        "#fbbf24",  # soft yellow
        "#34d399",  # soft green
        "#fca5a5",  # soft red
        "#a5b4fc",  # soft indigo
        "#f9fafb",  # very light gray
        "#cbd5e1",  # pale gray
        "#f472b6",  # soft pink
    ]
    return palette[:n_colors]

def calculate_lifestyle_score(profile, country):
    """Calculate lifestyle score based on profile and country data"""
    try:
        # Base score
        base_score = 50
        
        # Load life satisfaction data
        life_satisfaction_df = pd.read_csv('life_satisfaction_metrics.csv')
        life_data = life_satisfaction_df[life_satisfaction_df['Country'] == country]
        
        # Initialize scores
        work_life_score = 0
        social_score = 0
        culture_score = 0
        nature_score = 0
        
        if not life_data.empty:
            data = life_data.iloc[0]
            work_life_score = float(data['Work_Life_Balance_Score']) if not pd.isna(data['Work_Life_Balance_Score']) else 0
            social_score = float(data['Social_Life_Score']) if not pd.isna(data['Social_Life_Score']) else 0
            culture_score = float(data['Cultural_Activities_Score']) if not pd.isna(data['Cultural_Activities_Score']) else 0
            nature_score = float(data['Nature_Access_Score']) if not pd.isna(data['Nature_Access_Score']) else 0
        
        # Calculate weighted score
        total_score = (
            base_score +
            (work_life_score * 0.3) +
            (social_score * 0.2) +
            (culture_score * 0.2) +
            (nature_score * 0.3)
        )
        
        return min(100, max(0, total_score))
    except Exception as e:
        st.error(f"Error calculating lifestyle score: {e}")
        return 50

def clear_history():
    """Clear the analysis history from the database"""
    try:
        c.execute("DELETE FROM predictions")
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Database Error: {e}")
        return False

def get_popular_countries():
    """Get the most popular countries from the database"""
    try:
        c.execute("""
            SELECT predicted_country as Country, COUNT(*) as Count
            FROM predictions
            GROUP BY predicted_country
            ORDER BY Count DESC
            LIMIT 10
        """)
        return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Database Error: {e}")
        return []

def get_career_trends():
    """Get career field trends from the database"""
    try:
        c.execute("""
            SELECT career_field as 'Career Field', 
                   predicted_country as Country, 
                   COUNT(*) as Count
            FROM predictions
            GROUP BY career_field, predicted_country
            ORDER BY Count DESC
        """)
        return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Database Error: {e}")
        return []

def get_average_scores():
    """Get average scores for each country from the database"""
    try:
        c.execute("""
            SELECT predicted_country as Country,
                   AVG(risk_score) as Risk,
                   AVG(opportunity_score) as Opportunity,
                   AVG(lifestyle_score) as Lifestyle
            FROM predictions
            GROUP BY predicted_country
            ORDER BY (Risk + Opportunity + Lifestyle) / 3 DESC
        """)
        return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Database Error: {e}")
        return []

# --- Session State Initialization for Language/Culture Toggles ---
for country in ["Japan", "Germany"]:
    if f"show_lang_req_{country}" not in st.session_state:
        st.session_state[f"show_lang_req_{country}"] = False
    if f"show_culture_guide_{country}" not in st.session_state:
        st.session_state[f"show_culture_guide_{country}"] = False

# ------------------------- Helper Functions -------------------------
def load_external_data():
    """Load and process external data for analytics"""
    try:
        # Create separate DataFrames for different data types
        wb_data = pd.DataFrame.from_dict(EXTERNAL_DATA["World Bank"], orient='index')
        wb_data.columns = ['GDP Growth', 'Education Spending', 'Innovation Rank']
        
        hdi_data = pd.DataFrame.from_dict(EXTERNAL_DATA["UN HDI"], orient='index', columns=['HDI Score'])
        qs_data = pd.DataFrame.from_dict(EXTERNAL_DATA["QS University Rankings"], orient='index', columns=['Top Universities'])
        
        # Language requirements data
        lang_req_data = {
            'Japan': {
                'N1': {'Business': 35, 'IT/Tech': 20, 'Education': 50, 'Manufacturing': 40},
                'N2': {'Business': 65, 'IT/Tech': 40, 'Education': 80, 'Manufacturing': 70}
            },
            'Germany': {
                'C1': {'Business': 40, 'IT/Tech': 30, 'Education': 90, 'Manufacturing': 45},
                'B2': {'Business': 70, 'IT/Tech': 50, 'Education': 60, 'Manufacturing': 75}
            }
        }
        
        return {
            'world_bank': wb_data,
            'hdi': hdi_data,
            'qs_rankings': qs_data,
            'language_requirements': lang_req_data
        }
    except Exception as e:
        st.error(f"Error loading external data: {e}")
        return None

def process_uploaded_csv(uploaded_file):
    """Process uploaded CSV file and validate data"""
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Country', 'Score', 'Category', 'Language_Level']
        
        if not all(col in df.columns for col in required_columns):
            st.error("CSV must contain columns: Country, Score, Category, Language_Level")
            return None
        
        # Validate data types
        if not pd.to_numeric(df['Score'], errors='coerce').notnull().all():
            st.error("Score column must contain numeric values")
            return None
            
        # Validate score range
        if not ((df['Score'] >= 0) & (df['Score'] <= 100)).all():
            st.error("Scores must be between 0 and 100")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None

def analyze_part_time_work(country):
    """Analyze part-time work opportunities for a specific country"""
    try:
        # Get part-time work data for the country
        country_data = EXTERNAL_DATA["Part_Time_Work"].get(country, {})
        
        # Return formatted data
        return {
            'max_hours': country_data.get('max_hours', 'N/A'),
            'minimum_wage': country_data.get('minimum_wage_usd', 'N/A'),
            'student_allowed': country_data.get('student_allowed', False),
            'restrictions': country_data.get('restrictions', 'Information not available'),
            'popular_jobs': country_data.get('popular_jobs', []),
            'average_monthly_earning': country_data.get('average_monthly_earning', 'N/A')
        }
    except Exception as e:
        st.error(f"Error analyzing part-time work data: {e}")
        return {
            'max_hours': 'N/A',
            'minimum_wage': 'N/A',
            'student_allowed': False,
            'restrictions': 'Error retrieving data',
            'popular_jobs': [],
            'average_monthly_earning': 'N/A'
        }

def analyze_language_requirements(country, profile):
    """Analyze language requirements for a specific country"""
    career_field = profile['career_field']
    language_skills = profile['language_skills']
    
    # Map career fields to sectors
    career_mapping = {
        "Technology": "IT/Tech",
        "Education": "Education",
        "Engineering": "Manufacturing",
        "Business": "Business",
        "Finance": "Business",
        "Healthcare": "Business",
        "Arts & Media": "Business",
        "Science & Research": "Education"
    }
    
    sector = career_mapping.get(career_field, "Business")
    
    # Get language requirements for the sector
    if country == "Japan":
        requirements = {
            "IT/Tech": {"N2": 60, "N1": 40},
            "Education": {"N2": 80, "N1": 60},
            "Manufacturing": {"N2": 70, "N1": 50},
            "Business": {"N2": 75, "N1": 55}
        }
    else:
        requirements = EXTERNAL_DATA["Language_Requirements"][country][sector]
    
    # Calculate recommended language level
    current_level = language_skills
    recommended_level = "N2"  # Default recommendation
    if current_level < 5:
        recommended_level = "N5"
    elif current_level < 6:
        recommended_level = "N4"
    elif current_level < 7:
        recommended_level = "N3"
    elif current_level < 8:
        recommended_level = "N2"
    else:
        recommended_level = "N1"
    
    # JLPT study guide
    jlpt_guide = {
        "N5": {
            "description": "Basic Japanese knowledge",
            "study_materials": [
                "Genki I textbook",
                "Japanese From Zero! Book 1",
                "Tae Kim's Grammar Guide (Basic)",
                "Anki flashcards for N5 vocabulary"
            ],
            "practice_resources": [
                "JLPT N5 practice tests",
                "JapanesePod101 (Beginner)",
                "Duolingo Japanese course",
                "Memrise N5 vocabulary"
            ],
            "estimated_study_time": "3-4 months",
            "focus_areas": [
                "Basic grammar patterns",
                "Essential vocabulary (800 words)",
                "Hiragana and Katakana",
                "Basic kanji (100 characters)"
            ]
        },
        "N4": {
            "description": "Elementary Japanese knowledge",
            "study_materials": [
                "Genki II textbook",
                "Japanese From Zero! Book 2",
                "Tae Kim's Grammar Guide (Intermediate)",
                "Anki flashcards for N4 vocabulary"
            ],
            "practice_resources": [
                "JLPT N4 practice tests",
                "JapanesePod101 (Elementary)",
                "WaniKani (beginner levels)",
                "Bunpro grammar practice"
            ],
            "estimated_study_time": "4-6 months",
            "focus_areas": [
                "Intermediate grammar patterns",
                "Essential vocabulary (1,500 words)",
                "Basic kanji (300 characters)",
                "Reading comprehension"
            ]
        },
        "N3": {
            "description": "Intermediate Japanese knowledge",
            "study_materials": [
                "Tobira: Gateway to Advanced Japanese",
                "N3 Kanzen Master series",
                "Anki flashcards for N3 vocabulary",
                "Shin Kanzen Master N3"
            ],
            "practice_resources": [
                "JLPT N3 practice tests",
                "JapanesePod101 (Intermediate)",
                "WaniKani (intermediate levels)",
                "Bunpro N3 grammar"
            ],
            "estimated_study_time": "6-8 months",
            "focus_areas": [
                "Advanced grammar patterns",
                "Essential vocabulary (3,000 words)",
                "Intermediate kanji (650 characters)",
                "Business Japanese basics"
            ]
        },
        "N2": {
            "description": "Pre-advanced Japanese knowledge",
            "study_materials": [
                "N2 Kanzen Master series",
                "Shin Kanzen Master N2",
                "Anki flashcards for N2 vocabulary",
                "N2 Sou Matome series"
            ],
            "practice_resources": [
                "JLPT N2 practice tests",
                "JapanesePod101 (Upper Intermediate)",
                "WaniKani (advanced levels)",
                "Bunpro N2 grammar"
            ],
            "estimated_study_time": "8-12 months",
            "focus_areas": [
                "Advanced grammar patterns",
                "Essential vocabulary (6,000 words)",
                "Advanced kanji (1,000 characters)",
                "Business Japanese proficiency"
            ]
        },
        "N1": {
            "description": "Advanced Japanese knowledge",
            "study_materials": [
                "N1 Kanzen Master series",
                "Shin Kanzen Master N1",
                "Anki flashcards for N1 vocabulary",
                "N1 Sou Matome series"
            ],
            "practice_resources": [
                "JLPT N1 practice tests",
                "JapanesePod101 (Advanced)",
                "Native Japanese materials",
                "Bunpro N1 grammar"
            ],
            "estimated_study_time": "12-18 months",
            "focus_areas": [
                "Mastery of complex grammar",
                "Advanced vocabulary (10,000+ words)",
                "Complete kanji knowledge (2,000+ characters)",
                "Professional Japanese proficiency"
            ]
        }
    }
    
    # Language progression path
    progression_path = {
        "N5": {
            "duration": "3-4 months",
            "focus": "Basic communication, everyday phrases",
            "study_hours": "300-400"
        },
        "N4": {
            "duration": "4-6 months",
            "focus": "Basic grammar, common kanji",
            "study_hours": "400-500"
        },
        "N3": {
            "duration": "6-8 months",
            "focus": "Intermediate grammar, business basics",
            "study_hours": "500-600"
        },
        "N2": {
            "duration": "8-12 months",
            "focus": "Business communication, advanced grammar",
            "study_hours": "600-800"
        },
        "N1": {
            "duration": "12-18 months",
            "focus": "Native-level comprehension, specialized terminology",
            "study_hours": "800-1000"
        }
    }
    
    return {
        "sector": sector,
        "requirements": requirements.get(sector, requirements),
        "current_level": current_level,
        "recommended_level": recommended_level,
        "progression_path": progression_path,
        "jlpt_guide": jlpt_guide if country == "Japan" else None
    }

def display_language_culture_tips(country, profile):
    """Display language learning and work culture tips for a specific country"""
    if country in EXTERNAL_DATA["Language_Learning_Tips"]:
        tips = EXTERNAL_DATA["Language_Learning_Tips"][country]
        
        # Display language tips
        st.markdown(f"""
            <div style='background: rgba(30, 41, 59, 0.5);
                        padding: 1.5rem;
                        border-radius: 8px;
                        margin-top: 1rem;'>
                <h4 style='color: #60a5fa;'>Language Learning Path for {country}</h4>
                <ul style='color: white; list-style-type: none; padding-left: 0;'>
        """, unsafe_allow_html=True)
        
        for tip in tips['language_tips']:
            st.markdown(f"<li style='margin: 0.5rem 0; padding-left: 1.5rem; position: relative;'><span style='position: absolute; left: 0;'>•</span>{tip}</li>", unsafe_allow_html=True)
        
        # Display work culture tips
        st.markdown(f"""
                </ul>
                <h4 style='color: #60a5fa; margin-top: 1.5rem;'>Understanding {country} Work Culture</h4>
                <ul style='color: white; list-style-type: none; padding-left: 0;'>
        """, unsafe_allow_html=True)
        
        for tip in tips['work_culture_tips']:
            st.markdown(f"<li style='margin: 0.5rem 0; padding-left: 1.5rem; position: relative;'><span style='position: absolute; left: 0;'>•</span>{tip}</li>", unsafe_allow_html=True)
        
        # Display resources
        st.markdown("""
                </ul>
                <h4 style='color: #60a5fa; margin-top: 1.5rem;'>Additional Resources</h4>
                <ul style='color: white; list-style-type: none; padding-left: 0;'>
        """, unsafe_allow_html=True)
        
        for resource in tips['resources']:
            st.markdown(f"<li style='margin: 0.5rem 0; padding-left: 1.5rem; position: relative;'><span style='position: absolute; left: 0;'>•</span>{resource}</li>", unsafe_allow_html=True)
        
        st.markdown("""
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Display language progression path if available
        if 'language_progression' in tips:
            st.markdown(f"""
                <div style='background: rgba(30, 41, 59, 0.5);
                            padding: 1.5rem;
                            border-radius: 8px;
                            margin-top: 1rem;'>
                    <h4 style='color: #60a5fa;'>{country} Language Progression Path</h4>
                </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(len(tips['language_progression']))
            for idx, (level, details) in enumerate(tips['language_progression'].items()):
                with cols[idx]:
                    st.markdown(f"""
                        <div style='background: rgba(37, 99, 235, 0.1);
                                    padding: 1rem;
                                    border-radius: 6px;
                                    height: 100%;'>
                            <h5 style='color: #60a5fa; margin: 0;'>{level}</h5>
                            <p style='color: white; margin: 0.5rem 0;'><strong>Duration:</strong> {details['duration']}</p>
                            <p style='color: white; margin: 0.5rem 0;'><strong>Focus:</strong> {details['focus']}</p>
                            <p style='color: white; margin: 0.5rem 0;'><strong>Study Hours:</strong> {details['study_hours']}</p>
                        </div>
                    """, unsafe_allow_html=True)

def display_language_requirements(country, profile):
    """Display language requirements and study guide"""
    try:
        language_analysis = analyze_language_requirements(country, profile)
        
        st.markdown("### Language Requirements Analysis")
        st.write(f"**Current Level:** {language_analysis['current_level']}/10")
        st.write(f"**Recommended Level:** {language_analysis['recommended_level']}")
        st.write(f"**Estimated Study Duration:** {language_analysis['progression_path'][language_analysis['recommended_level']]['duration']}")
        st.write(f"**Focus Areas:** {', '.join(language_analysis['progression_path'][language_analysis['recommended_level']]['focus'].split(', '))}")

        # Language Progression Path
        st.markdown("### Language Progression Path")
        levels = list(language_analysis['progression_path'].keys())
        current_index = levels.index(language_analysis['recommended_level'])

        # Create a visual progression path
        progress_cols = st.columns(len(levels))
        for i, level in enumerate(levels):
            with progress_cols[i]:
                if i <= current_index:
                    st.markdown(f"**{level}** ✅")
                else:
                    st.markdown(f"**{level}**")
                st.write(f"Duration: {language_analysis['progression_path'][level]['duration']}")
                st.write(f"Hours: {language_analysis['progression_path'][level]['study_hours']}")

        # Language Level Guide
        st.markdown("### Language Level Guide")
        if country == "Japan":
            st.markdown("""
                <div style='background: rgba(30, 41, 59, 0.5);
                            padding: 1.5rem;
                            border-radius: 8px;
                            margin: 1rem 0;'>
                    <h4 style='color: #60a5fa;'>Japanese (JLPT)</h4>
                    <ul style='color: white; list-style-type: none; padding-left: 0;'>
                        <li style='margin: 0.5rem 0;'><strong>N1:</strong> Advanced level - Can understand Japanese in any situation</li>
                        <li style='margin: 0.5rem 0;'><strong>N2:</strong> Upper-intermediate - Business level Japanese</li>
                        <li style='margin: 0.5rem 0;'><strong>N3:</strong> Intermediate - Can understand everyday Japanese</li>
                        <li style='margin: 0.5rem 0;'><strong>N4:</strong> Elementary - Basic Japanese knowledge</li>
                        <li style='margin: 0.5rem 0;'><strong>N5:</strong> Beginner - Basic Japanese knowledge</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        elif country == "Germany":
            st.markdown("""
                <div style='background: rgba(30, 41, 59, 0.5);
                            padding: 1.5rem;
                            border-radius: 8px;
                            margin: 1rem 0;'>
                    <h4 style='color: #60a5fa;'>German (CEFR)</h4>
                    <ul style='color: white; list-style-type: none; padding-left: 0;'>
                        <li style='margin: 0.5rem 0;'><strong>C2:</strong> Mastery level - Native-like proficiency</li>
                        <li style='margin: 0.5rem 0;'><strong>C1:</strong> Advanced level - Professional working proficiency</li>
                        <li style='margin: 0.5rem 0;'><strong>B2:</strong> Upper-intermediate - Can work in German-speaking environment</li>
                        <li style='margin: 0.5rem 0;'><strong>B1:</strong> Intermediate - Can handle most situations</li>
                        <li style='margin: 0.5rem 0;'><strong>A2:</strong> Elementary - Basic communication skills</li>
                        <li style='margin: 0.5rem 0;'><strong>A1:</strong> Beginner - Basic German knowledge</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Study Guide
        if country == "Japan" and language_analysis['jlpt_guide']:
            st.markdown("### JLPT Study Guide")
            jlpt_tabs = st.tabs(levels)
            for i, level in enumerate(levels):
                with jlpt_tabs[i]:
                    guide = language_analysis['jlpt_guide'][level]
                    st.markdown(f"**Description:** {guide['description']}")
                    
                    st.markdown("**Study Materials:**")
                    for material in guide['study_materials']:
                        st.markdown(f"- {material}")
                    
                    st.markdown("**Practice Resources:**")
                    for resource in guide['practice_resources']:
                        st.markdown(f"- {resource}")
                    
                    st.markdown(f"**Estimated Study Time:** {guide['estimated_study_time']}")
                    
                    st.markdown("**Focus Areas:**")
                    for area in guide['focus_areas']:
                        st.markdown(f"- {area}")
        elif country == "Germany":
            st.markdown("### German Language Study Guide")
            cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
            cefr_tabs = st.tabs(cefr_levels)
            
            cefr_guide = {
                "A1": {
                    "description": "Beginner level - Basic communication skills",
                    "study_materials": [
                        "Deutsch im Einsatz",
                        "Menschen A1",
                        "Studio 21 A1",
                        "Basic German vocabulary flashcards"
                    ],
                    "practice_resources": [
                        "Deutsche Welle A1 course",
                        "Goethe-Institut A1 materials",
                        "Duolingo German course",
                        "Memrise A1 vocabulary"
                    ],
                    "estimated_study_time": "2-3 months",
                    "focus_areas": [
                        "Basic greetings and introductions",
                        "Simple present tense",
                        "Essential vocabulary (500 words)",
                        "Basic pronunciation"
                    ]
                },
                "A2": {
                    "description": "Elementary level - Basic social interaction",
                    "study_materials": [
                        "Deutsch im Einsatz A2",
                        "Menschen A2",
                        "Studio 21 A2",
                        "A2 grammar workbook"
                    ],
                    "practice_resources": [
                        "Deutsche Welle A2 course",
                        "Goethe-Institut A2 materials",
                        "Lingoda A2 classes",
                        "GermanPod101 (Elementary)"
                    ],
                    "estimated_study_time": "3-4 months",
                    "focus_areas": [
                        "Past tense basics",
                        "Everyday conversations",
                        "Essential vocabulary (1,000 words)",
                        "Basic writing skills"
                    ]
                },
                "B1": {
                    "description": "Intermediate level - Independent language use",
                    "study_materials": [
                        "Deutsch im Einsatz B1",
                        "Menschen B1",
                        "Studio 21 B1",
                        "B1 grammar workbook"
                    ],
                    "practice_resources": [
                        "Deutsche Welle B1 course",
                        "Goethe-Institut B1 materials",
                        "Lingoda B1 classes",
                        "GermanPod101 (Intermediate)"
                    ],
                    "estimated_study_time": "4-6 months",
                    "focus_areas": [
                        "Complex grammar structures",
                        "Business German basics",
                        "Essential vocabulary (2,000 words)",
                        "Reading comprehension"
                    ]
                },
                "B2": {
                    "description": "Upper-intermediate level - Professional working proficiency",
                    "study_materials": [
                        "Deutsch im Einsatz B2",
                        "Menschen B2",
                        "Studio 21 B2",
                        "B2 grammar workbook"
                    ],
                    "practice_resources": [
                        "Deutsche Welle B2 course",
                        "Goethe-Institut B2 materials",
                        "Lingoda B2 classes",
                        "GermanPod101 (Upper Intermediate)"
                    ],
                    "estimated_study_time": "6-8 months",
                    "focus_areas": [
                        "Advanced grammar",
                        "Business communication",
                        "Essential vocabulary (4,000 words)",
                        "Writing professional emails"
                    ]
                },
                "C1": {
                    "description": "Advanced level - Professional working proficiency",
                    "study_materials": [
                        "Deutsch im Einsatz C1",
                        "Menschen C1",
                        "Studio 21 C1",
                        "C1 grammar workbook"
                    ],
                    "practice_resources": [
                        "Deutsche Welle C1 course",
                        "Goethe-Institut C1 materials",
                        "Lingoda C1 classes",
                        "GermanPod101 (Advanced)"
                    ],
                    "estimated_study_time": "8-12 months",
                    "focus_areas": [
                        "Complex grammar mastery",
                        "Professional presentations",
                        "Essential vocabulary (6,000 words)",
                        "Academic writing"
                    ]
                },
                "C2": {
                    "description": "Mastery level - Native-like proficiency",
                    "study_materials": [
                        "Deutsch im Einsatz C2",
                        "Menschen C2",
                        "Studio 21 C2",
                        "C2 grammar workbook"
                    ],
                    "practice_resources": [
                        "Deutsche Welle C2 course",
                        "Goethe-Institut C2 materials",
                        "Lingoda C2 classes",
                        "Native German materials"
                    ],
                    "estimated_study_time": "12-18 months",
                    "focus_areas": [
                        "Perfect grammar mastery",
                        "Professional negotiation",
                        "Complete vocabulary (8,000+ words)",
                        "Academic research"
                    ]
                }
            }
            
            for i, level in enumerate(cefr_levels):
                with cefr_tabs[i]:
                    guide = cefr_guide[level]
                    st.markdown(f"**Description:** {guide['description']}")
                    
                    st.markdown("**Study Materials:**")
                    for material in guide['study_materials']:
                        st.markdown(f"- {material}")
                    
                    st.markdown("**Practice Resources:**")
                    for resource in guide['practice_resources']:
                        st.markdown(f"- {resource}")
                    
                    st.markdown(f"**Estimated Study Time:** {guide['estimated_study_time']}")
                    
                    st.markdown("**Focus Areas:**")
                    for area in guide['focus_areas']:
                        st.markdown(f"- {area}")

        # Language Learning Tips
        st.markdown("### Language Learning Tips")
        tips = EXTERNAL_DATA["Language_Learning_Tips"][country]["tips"]
        for tip in tips:
            st.markdown(f"- {tip}")
            
    except Exception as e:
        st.error(f"Error displaying language requirements: {e}")

# Constants and Templates
EXAMPLE_CSV = """Country,Score,Category,Language_Level
Japan,85,IT/Tech,N2
Japan,90,Business,N1
Germany,80,IT/Tech,B2
Germany,88,Education,C1
"""

PART_TIME_CSV = """Country,Hours_Per_Week,Minimum_Wage_USD,Student_Allowed,Visa_Required
Japan,28,8.50,Yes,Yes
Germany,20,12.00,Yes,Yes
USA,20,7.25,Yes,Yes
Canada,20,12.50,Yes,Yes
UK,20,11.80,Yes,Yes
Australia,40,15.20,Yes,Yes
Singapore,20,No minimum,Yes,Yes
Netherlands,16,11.50,Yes,Yes
Sweden,20,No minimum,Yes,Yes
Switzerland,15,No minimum,Yes,Yes
Ireland,20,11.30,Yes,Yes
"""

# Must be the first Streamlit command
st.set_page_config(
    page_title="PathWise – Global Life Decision Assistant",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables for external data sources
EXTERNAL_DATA = {
    "World Bank": {
        "USA": {"gdp_growth": 5.7, "education_spend": 5.0, "innovation_rank": 3},
        "Canada": {"gdp_growth": 4.6, "education_spend": 5.3, "innovation_rank": 16},
        "UK": {"gdp_growth": 7.4, "education_spend": 5.5, "innovation_rank": 4},
        "Australia": {"gdp_growth": 4.7, "education_spend": 5.1, "innovation_rank": 25},
        "Germany": {"gdp_growth": 2.9, "education_spend": 4.9, "innovation_rank": 8},
        "Japan": {"gdp_growth": 1.6, "education_spend": 3.1, "innovation_rank": 13},
        "Singapore": {"gdp_growth": 7.6, "education_spend": 2.9, "innovation_rank": 7}
    },
    "UN HDI": {
        "USA": 0.921,
        "Canada": 0.929,
        "UK": 0.932,
        "Australia": 0.944,
        "Germany": 0.942,
        "Japan": 0.919,
        "Singapore": 0.938
    },
    "QS University Rankings": {
        "USA": 30,
        "Canada": 7,
        "UK": 18,
        "Australia": 8,
        "Germany": 12,
        "Japan": 5,
        "Singapore": 2
    },
    "Language_Requirements": {
        "Japan": {
            "Business": {"JLPT_N2": 65, "JLPT_N1": 35},
            "IT/Tech": {"JLPT_N2": 40, "JLPT_N1": 20},
            "Education": {"JLPT_N2": 80, "JLPT_N1": 50},
            "Manufacturing": {"JLPT_N2": 70, "JLPT_N1": 40}
        },
        "Germany": {
            "Business": {"B2": 70, "C1": 40},
            "IT/Tech": {"B1": 50, "B2": 30},
            "Education": {"C1": 90, "C2": 60},
            "Manufacturing": {"B2": 75, "C1": 45}
        }
    },
    "Work_Culture": {
        "Japan": {
            "Overtime_Hours": 25,
            "Stress_Level": 7.2,
            "Work_Life_Balance": 5.5
        },
        "Germany": {
            "Overtime_Hours": 10,
            "Stress_Level": 5.8,
            "Work_Life_Balance": 7.5
        }
    },
    "Language_Schools": {
        "Japan": {
            "Tokyo": 150,
            "Osaka": 80,
            "Kyoto": 45,
            "Fukuoka": 35
        },
        "Germany": {
            "Berlin": 120,
            "Munich": 85,
            "Hamburg": 55,
            "Frankfurt": 50
        }
    },
    "Part_Time_Work": {
        "Japan": {
            "max_hours": 28,
            "minimum_wage_usd": 8.50,
            "student_allowed": True,
            "restrictions": "Need valid student/work visa",
            "popular_jobs": ["Convenience Store", "English Teaching", "Restaurant", "Retail"],
            "average_monthly_earning": 800
        },
        "Germany": {
            "max_hours": 20,
            "minimum_wage_usd": 12.00,
            "student_allowed": True,
            "restrictions": "EU citizens or valid permit required",
            "popular_jobs": ["Cafe/Restaurant", "University Assistant", "Retail", "Delivery"],
            "average_monthly_earning": 950
        },
        "USA": {
            "max_hours": 20,
            "minimum_wage_usd": 7.25,
            "student_allowed": True,
            "restrictions": "Valid F-1 visa required",
            "popular_jobs": ["Campus Work", "Retail", "Food Service", "Tutoring"],
            "average_monthly_earning": 800
        },
        "Canada": {
            "max_hours": 20,
            "minimum_wage_usd": 12.50,
            "student_allowed": True,
            "restrictions": "Study permit required",
            "popular_jobs": ["Retail", "Food Service", "Campus Work", "Customer Service"],
            "average_monthly_earning": 1000
        },
        "UK": {
            "max_hours": 20,
            "minimum_wage_usd": 11.80,
            "student_allowed": True,
            "restrictions": "Tier 4 visa holders allowed",
            "popular_jobs": ["Retail", "Hospitality", "Admin", "Student Ambassador"],
            "average_monthly_earning": 900
        },
        "Australia": {
            "max_hours": 40,
            "minimum_wage_usd": 15.20,
            "student_allowed": True
        },
        "Singapore": {
            "max_hours": 20,
            "minimum_wage_usd": 15.20,
            "student_allowed": True,
            "restrictions": "No minimum wage",
            "popular_jobs": ["Retail", "Food Service", "Campus Work", "Customer Service"],
            "average_monthly_earning": 1000
        },
        "Netherlands": {
            "max_hours": 16,
            "minimum_wage_usd": 11.50,
            "student_allowed": True,
            "restrictions": "EU citizens or valid permit required",
            "popular_jobs": ["Retail", "Food Service", "Campus Work", "Customer Service"],
            "average_monthly_earning": 950
        },
        "Sweden": {
            "max_hours": 20,
            "minimum_wage_usd": 15.20,
            "student_allowed": True,
            "restrictions": "EU citizens or valid permit required",
            "popular_jobs": ["Retail", "Food Service", "Campus Work", "Customer Service"],
            "average_monthly_earning": 1000
        },
        "Switzerland": {
            "max_hours": 15,
            "minimum_wage_usd": 15.20,
            "student_allowed": True,
            "restrictions": "EU citizens or valid permit required",
            "popular_jobs": ["Retail", "Food Service", "Campus Work", "Customer Service"],
            "average_monthly_earning": 1000
        },
        "Ireland": {
            "max_hours": 20,
            "minimum_wage_usd": 11.30,
            "student_allowed": True,
            "restrictions": "EU citizens or valid permit required",
            "popular_jobs": ["Retail", "Food Service", "Campus Work", "Customer Service"],
            "average_monthly_earning": 950
        }
    }
}

# Global variables for country data
tuition_factors = {
    "USA": 0.5,        # Very high tuition costs
    "Canada": 0.7,     # High costs
    "UK": 0.6,         # High costs
    "Australia": 0.65, # High costs
    "Germany": 1.4,    # Very affordable (often free)
    "Japan": 0.8,      # Moderate to high costs
    "Singapore": 0.7,  # High costs
    "New Zealand": 0.7,# High costs
    "Netherlands": 1.0,# More affordable
    "Sweden": 1.4,     # Very affordable (often free)
    "Switzerland": 0.8,# High costs
    "Ireland": 0.7     # High costs
}

def get_country_recommendations(country, profile):
    recommendations = {
        "USA": {
            "Tech": "Leading tech hubs in Silicon Valley, Seattle, and Austin offer cutting-edge opportunities.",
            "Education": "Home to world-renowned universities like MIT, Stanford, and Harvard.",
            "Lifestyle": "Diverse culture, high salaries, but consider healthcare costs.",
            "Innovation": "Strong startup ecosystem and venture capital funding."
        },
        "Canada": {
            "Tech": "Growing tech scene in Toronto, Vancouver, and Montreal.",
            "Education": "High-quality education with more affordable tuition than the USA.",
            "Lifestyle": "High quality of life, universal healthcare, and welcoming immigration policies.",
            "Innovation": "Strong government support for innovation and research."
        },
        "UK": {
            "Tech": "London's fintech scene and emerging tech hubs across the country.",
            "Education": "Historic universities with strong global recognition.",
            "Lifestyle": "Rich cultural heritage and access to European travel.",
            "Innovation": "Strong focus on AI and financial innovation."
        },
        "Australia": {
            "Tech": "Growing tech sector in Sydney and Melbourne.",
            "Education": "High-quality education system with strong research focus.",
            "Lifestyle": "Excellent work-life balance and outdoor lifestyle.",
            "Innovation": "Strong focus on renewable energy and sustainability."
        },
        "Germany": {
            "Tech": "Strong manufacturing and Industry 4.0 focus.",
            "Education": "Tuition-free universities and strong vocational training.",
            "Lifestyle": "High social security and work benefits.",
            "Innovation": "Leader in automotive and engineering innovation."
        },
        "Japan": {
            "Tech": "Robotics and electronics innovation hub.",
            "Education": "Strong emphasis on research and development.",
            "Lifestyle": "Unique culture and excellent public infrastructure.",
            "Innovation": "Leading in robotics and automation."
        },
        "Singapore": {
            "Tech": "Smart nation initiatives and strong digital infrastructure.",
            "Education": "Education hub with global university partnerships.",
            "Lifestyle": "Safe, clean, and efficient city-state.",
            "Innovation": "Strong government support for digital transformation."
        }
    }
    return recommendations.get(country, {
        "Tech": "Growing technology sector.",
        "Education": "Quality education system.",
        "Lifestyle": "Balanced quality of life.",
        "Innovation": "Developing innovation ecosystem."
    })

def calculate_risk_score(profile, country):
    try:
        # Factors that contribute to higher risk. Scale is 0-1.
        # Higher value in these factors means higher country-specific risk.
        country_inherent_risk_factors = {
            "USA": {"stability": 0.3, "safety": 0.25, "healthcare_access": 0.4}, # Example values
            "Canada": {"stability": 0.2, "safety": 0.2, "healthcare_access": 0.1}, # Example values
            "UK": {"stability": 0.35, "safety": 0.3, "healthcare_access": 0.2}, # Example values
            "Australia": {"stability": 0.2, "safety": 0.2, "healthcare_access": 0.15}, # Example values
            "Germany": {"stability": 0.15, "safety": 0.15, "healthcare_access": 0.05}, # Example values
            "Japan": {"stability": 0.2, "safety": 0.1, "healthcare_access": 0.1}, # Example values
            "Singapore": {"stability": 0.1, "safety": 0.05, "healthcare_access": 0.1}, # Example values
            "New Zealand": {"stability": 0.15, "safety": 0.1, "healthcare_access": 0.1}, # Example values
            "Netherlands": {"stability": 0.2, "safety": 0.15, "healthcare_access": 0.1}, # Example values
            "Sweden": {"stability": 0.2, "safety": 0.2, "healthcare_access": 0.05}, # Example values
            "Switzerland": {"stability": 0.1, "safety": 0.1, "healthcare_access": 0.05}, # Example values
            "Ireland": {"stability": 0.3, "safety": 0.2, "healthcare_access": 0.25} # Example values
        }

        # User-specific factors contributing to risk
        # Lower language skills -> higher risk (0=high risk, 10=low risk -> invert)
        language_risk = 1 - (profile['language_skills'] / 10)
        # Lower savings -> higher risk (scale savings to a factor, inverse relationship)
        savings_risk = 1 - min(1, profile['savings'] / 50000) # Assuming 50k savings significantly reduces financial risk
        # User risk tolerance: High tolerance might mean they perceive less risk, but the actual risk could still be high.
        # Let's make this neutral for now or a minor factor of how user *perceives* it.
        user_perception_factor = (10 - profile['risk_tolerance']) / 10 # Lower tolerance -> higher perceived risk impact

        # Get country-specific inherent risk (average of its factors)
        country_risk_values = country_inherent_risk_factors.get(country, {"stability": 0.3, "safety": 0.3, "healthcare_access": 0.3})
        inherent_country_risk = (country_risk_values["stability"] + country_risk_values["safety"] + country_risk_values["healthcare_access"]) / 3

        # Calculate total raw risk score by combining factors
        # Weights need adjustment based on desired impact
        raw_risk_score = (
            language_risk * 30 +          # Language risk has a significant impact
            savings_risk * 40 +           # Financial risk has a significant impact
            inherent_country_risk * 30   # Country's inherent risk
            # user_perception_factor * 10 # Minor impact based on user's tolerance
        )

        # Scale raw score to 0-100 range
        # Max possible raw_risk_score (if all risk factors are 1) is 30+40+30 = 100
        # Min possible raw_risk_score (if all risk factors are 0) is 0
        final_risk_score = min(100, max(0, raw_risk_score))

        return final_risk_score
    except Exception as e:
        st.error(f"Error calculating risk score: {e}")
        return 50 # Return a moderate risk score in case of error

def calculate_opportunity_score(profile, country):
    try:
        # Drastically lower base score
        base_score = 15  # Further reduced from 20
        
        # Load life satisfaction data
        life_satisfaction_df = pd.read_csv('life_satisfaction_metrics.csv')
        life_data = life_satisfaction_df[life_satisfaction_df['Country'] == country]
        
        # Add life satisfaction bonus if country data exists
        life_satisfaction_bonus = 0
        work_life_bonus = 0
        social_bonus = 0
        
        if not life_data.empty:
            data = life_data.iloc[0]
            # Handle NaN values
            life_satisfaction_bonus = (float(data['Life_Satisfaction_Score']) / 10) * 5 if not pd.isna(data['Life_Satisfaction_Score']) else 0
            work_life_bonus = (float(data['Work_Life_Balance_Score']) / 10) * 3 if not pd.isna(data['Work_Life_Balance_Score']) else 0
            social_bonus = (float(data['Social_Life_Score']) / 10) * 2 if not pd.isna(data['Social_Life_Score']) else 0
        
        # Career factors with extreme differentiation for technology
        career_factors = {
            "Technology": {
                "USA": 1.6, "Canada": 1.0, "UK": 1.1, "Australia": 0.8,
                "Germany": 1.0, "Japan": 1.1, "Singapore": 1.05, "Switzerland": 0.9,
                "Netherlands": 0.8, "Sweden": 0.9, "Ireland": 0.8, "New Zealand": 0.7
            }
        }
        
        # Very aggressive career multiplier fallback
        career_multiplier = career_factors.get(profile['career_field'], {}).get(country, 0.6)
        
        # Drastically reduced education base multipliers
        education_base_multipliers = {
            "High School": 0.4,    # Further reduced
            "Bachelor's": 0.6,     # Further reduced
            "Master's": 0.8,       # Further reduced
            "PhD": 1.0             # Further reduced
        }
        
        # More extreme education country modifiers
        education_country_modifier = {
            "USA": 1.3, "UK": 1.2, "Germany": 1.3, "Australia": 0.9,
            "Canada": 0.95, "Japan": 1.1, "Switzerland": 1.2, "Singapore": 1.15,
            "Netherlands": 1.0, "Sweden": 1.2, "Ireland": 0.9, "New Zealand": 0.85
        }.get(country, 0.7)  # Further reduced default
        
        education_multiplier = education_base_multipliers[profile['education']] * education_country_modifier
        
        # Calculate base impacts with more aggressive reductions
        career_base = (career_multiplier * 20) - 8
        education_base = (education_multiplier * 15) - 6
        language_base = ((profile['language_skills'] / 10) * 12) - 4
        tuition_base = (tuition_factors.get(country, 0.7) * (profile['tuition_importance'] / 10) * 10) - 3
        
        # Apply scaling factors
        career_impact = career_base * 0.8
        education_impact = education_base * 0.7
        language_impact = language_base * 0.6
        tuition_impact = tuition_base * 0.7
        
        # Very aggressive age impact
        age_penalty = max(0, (profile['age'] - 25) * 1.2)
        age_impact = max(0, 6 - age_penalty)
        
        # Highly restrictive market impact
        market_impact = min(5, (profile['savings'] / 150000) * 4)
        
        # Add life satisfaction impacts
        total_life_impact = life_satisfaction_bonus + work_life_bonus + social_bonus
        
        # Store components for debugging
        score_components = {
            'base_score': base_score,
            'career_impact': career_impact,
            'education_impact': education_impact,
            'language_impact': language_impact,
            'tuition_impact': tuition_base,
            'age_impact': age_impact,
            'market_impact': market_impact,
            'life_satisfaction_impact': total_life_impact,
            'multipliers': {
                'career': career_multiplier,
                'education': education_multiplier,
                'tuition': tuition_factors.get(country, 0.7)
            }
        }
        
        # Calculate preliminary score
        total_score = (
            base_score +
            career_impact +
            education_impact +
            language_impact +
            tuition_impact +
            age_impact +
            market_impact +
            total_life_impact  # Add life satisfaction impact
        )
        
        # Apply harsh country-specific penalties
        country_penalties = {
            "USA": 0.85,     # Extreme competition
            "Canada": 0.8,   # Weather and high living costs
            "UK": 0.8,       # Brexit and market uncertainty
            "Australia": 0.75, # Geographic isolation
            "Germany": 0.8,   # Significant language barrier
            "Japan": 0.7,     # Extreme cultural/language barriers
            "Singapore": 0.75, # Very high cost of living
            "New Zealand": 0.7, # Limited market
            "Netherlands": 0.75, # Language and market size
            "Sweden": 0.75,    # Climate and language
            "Switzerland": 0.8, # Extreme cost of living
            "Ireland": 0.75    # Limited market size
        }
        
        # Apply multiple reduction factors
        penalty = country_penalties.get(country, 0.7)
        preliminary_score = total_score * penalty
        
        # Apply final scaling and caps
        final_score = min(85, max(20, preliminary_score * 0.8))
        
        return final_score, score_components
        
    except Exception as e:
        st.error(f"Error calculating opportunity score: {e}")
        return 15, {}

# Add this to the country analysis section after recommendations
def display_life_satisfaction_metrics(country):
    try:
        # Load life satisfaction data
        life_satisfaction_df = pd.read_csv('life_satisfaction_metrics.csv')
        life_data = life_satisfaction_df[life_satisfaction_df['Country'] == country]
        
        if not life_data.empty:
            data = life_data.iloc[0]
            
            # Convert numeric columns to float and handle NaN values
            numeric_columns = [
                'Life_Satisfaction_Score', 'Work_Life_Balance_Score', 
                'Mental_Health_Index', 'Social_Life_Score', 
                'Cultural_Activities_Score', 'Nature_Access_Score'
            ]
            
            for col in numeric_columns:
                if pd.isna(data[col]):
                    data[col] = 0.0
                else:
                    data[col] = float(data[col])
            
            # Convert integer columns and handle NaN values
            integer_columns = [
                'World_Happiness_Rank', 'Paid_Vacation_Days', 'Public_Holidays'
            ]
            
            for col in integer_columns:
                if pd.isna(data[col]):
                    data[col] = 0
                else:
                    data[col] = int(float(data[col]))
            
            # --- Modern Card for Life Satisfaction Metrics ---
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #2563eb 0%, #1e293b 100%);
                            padding: 1.7rem 1.2rem 1.2rem 1.2rem;
                            border-radius: 16px;
                            margin: 1.5rem 0;
                            box-shadow: 0 6px 24px rgba(59,130,246,0.10);'>
                    <h4 style='color: #60a5fa; margin-bottom: 1.2rem;'>🌟 Quality of Life Metrics for {country}</h4>
                    <div style='display: flex; gap: 2rem; flex-wrap: wrap;'>
                        <div style='flex: 1; min-width: 180px; background: rgba(37, 99, 235, 0.13); padding: 1.1rem; border-radius: 10px; margin-bottom: 1rem;'>
                            <span style='color: #fbbf24; font-size: 1.1rem;'>💼 Work-Life Balance</span>
                            <div style='color: white; font-size: 1.5rem; font-weight: 600;'>{data['Work_Life_Balance_Score']:.1f}/10</div>
                        </div>
                        <div style='flex: 1; min-width: 180px; background: rgba(16, 185, 129, 0.13); padding: 1.1rem; border-radius: 10px; margin-bottom: 1rem;'>
                            <span style='color: #38bdf8; font-size: 1.1rem;'>🎭 Social & Culture</span>
                            <div style='color: white; font-size: 1.5rem; font-weight: 600;'>{data['Social_Life_Score']:.1f}/10</div>
                            <div style='color: #a5b4fc; font-size: 1.05rem;'>Culture Score: {data['Cultural_Activities_Score']:.1f}/10</div>
                        </div>
                    </div>
                    <div style='margin-top: 1.2rem; background: rgba(59,130,246,0.07); padding: 1.1rem; border-radius: 10px;'>
                        <p style='color: #fbbf24; font-size: 1.1rem; margin-bottom: 0.5rem;'>🌟 Key Benefits:</p>
                        <ul style='color: white; margin: 0 0 0 1.2rem; font-size: 1.08rem;'>
                            <li>🗓️ <b>Total Leave:</b> {data['Paid_Vacation_Days']} paid leave + {data['Public_Holidays']} holidays</li>
                            <li>🧠 <b>Mental Health Index:</b> {data['Mental_Health_Index']:.1f}/100</li>
                            <li>🌳 <b>Nature Access Score:</b> {data['Nature_Access_Score']:.1f}/10</li>
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- CSV Visualizations under Expanders ---
            with st.expander("🏫 Show Top Universities"):
                try:
                    university_rankings = {
                        "Japan": ["University of Tokyo", "Kyoto University", "Osaka University", "Tokyo Institute of Technology", "Tohoku University"],
                        "Germany": ["Technical University of Munich", "Heidelberg University", "Ludwig Maximilian University of Munich", "Humboldt University of Berlin", "RWTH Aachen University"],
                        "USA": ["Massachusetts Institute of Technology", "Stanford University", "Harvard University", "California Institute of Technology", "University of Chicago"],
                        "UK": ["University of Oxford", "University of Cambridge", "Imperial College London", "University College London", "University of Edinburgh"],
                        "Canada": ["University of Toronto", "University of British Columbia", "McGill University", "University of Montreal", "University of Alberta"],
                        "Australia": ["University of Melbourne", "University of Sydney", "Australian National University", "University of Queensland", "University of New South Wales"],
                        "Singapore": ["National University of Singapore", "Nanyang Technological University", "Singapore Management University", "Singapore University of Technology and Design", "Singapore Institute of Technology"]
                    }
                    if country in university_rankings:
                        st.markdown("<ul style='color: white;'>", unsafe_allow_html=True)
                        for i, university in enumerate(university_rankings[country], 1):
                            st.markdown(f"<li style='margin: 0.5rem 0;'>{i}. {university}</li>", unsafe_allow_html=True)
                        st.markdown("</ul>", unsafe_allow_html=True)
                    else:
                        uni_data = pd.read_csv('university_rankings.csv')
                        country_unis = uni_data[uni_data['Country'] == country].head(5)
                        if not country_unis.empty:
                            fig = px.bar(country_unis, x='University', y='Rank', title=f'Top Universities in {country}',
                                         labels={'Rank': 'Global Rank'}, color='Rank', color_continuous_scale='Viridis')
                            fig.update_layout(
                                plot_bgcolor='rgba(30, 41, 59, 0.3)',
                                paper_bgcolor='rgba(30, 41, 59, 0.3)',
                                font_color='white',
                                title_font_color='#60a5fa',
                                title_font_size=20,
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No university ranking data available for {country}")
                except Exception as e:
                    st.info(f"University ranking data not available: {e}")

            # --- Tuition & Part-Time Work Expander ---
            with st.expander("💸 Show Tuition & Part-Time Work Info", expanded=False):
                try:
                    tuition_df = pd.read_csv('tuition_part_time.csv')
                    row = tuition_df[tuition_df['Country'].str.lower() == country.lower()]
                    if not row.empty:
                        row = row.iloc[0]
                        st.markdown(f"""
                        <div style='background:#232a36; color:#f4f4f9; border-radius:10px; padding:1rem; margin:1rem 0;'>
                            <b>Tuition & Part-Time Work in {country}</b><br>
                            <ul style='margin:0.5rem 0 0 1.2rem; color:#f4f4f9;'>
                                <li><b>Tuition Cost:</b> ${row['Tuition Cost (USD)']:,}</li>
                                <li><b>Part-Time Hour Limit:</b> {row['Part-Time Hour Limit']} hours/week</li>
                                <li><b>Min. Salary per Hour:</b> ${row['Part-Time Pay Per Hour (USD)']:.2f}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"No tuition/part-time data available for {country}.")
                except Exception as e:
                    st.info(f"Error loading tuition/part-time data: {e}")

            with st.expander("📊 Show Quality of Life Metrics"):
                try:
                    metrics = {
                        'Metric': ['Life Satisfaction', 'Work-Life Balance', 'Social Life', 'Cultural Activities', 'Nature Access'],
                        'Score': [
                            float(data['Life_Satisfaction_Score']),
                            float(data['Work_Life_Balance_Score']),
                            float(data['Social_Life_Score']),
                            float(data['Cultural_Activities_Score']),
                            float(data['Nature_Access_Score'])
                        ]
                    }
                    metrics_df = pd.DataFrame(metrics)
                    fig = px.bar(metrics_df, x='Metric', y='Score', title=f'Quality of Life Metrics for {country}',
                                 labels={'Score': 'Score (out of 10)'}, color='Score', color_continuous_scale='Viridis')
                    fig.update_layout(
                        plot_bgcolor='rgba(30, 41, 59, 0.3)',
                        paper_bgcolor='rgba(30, 41, 59, 0.3)',
                        font_color='white',
                        title_font_color='#60a5fa',
                        title_font_size=18, # Slightly reduced title font size
                        showlegend=False,
                        height=280, # Reduced height
                        margin=dict(t=40, b=10, l=10, r=10) # Adjusted margins
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Quality of life metrics data not available: {e}")

            with st.expander("💹 Show Economic Indicators"):
                try:
                    gdp_data = pd.read_csv('gdp_data.csv')
                    country_gdp = gdp_data[gdp_data['Country'] == country]
                    if not country_gdp.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=country_gdp['Year'],
                            y=country_gdp['GDP_Growth'],
                            name='GDP Growth',
                            line=dict(color='#60a5fa', width=3)
                        ))
                        if 'HDI' in country_gdp.columns:
                            fig.add_trace(go.Scatter(
                                x=country_gdp['Year'],
                                y=country_gdp['HDI'],
                                name='HDI',
                                line=dict(color='#34d399', width=3)
                            ))
                        fig.update_layout(
                            title=f'Economic Indicators for {country}',
                            xaxis_title='Year',
                            yaxis_title='Value',
                            plot_bgcolor='rgba(30, 41, 59, 0.3)',
                            paper_bgcolor='rgba(30, 41, 59, 0.3)',
                            font_color='white',
                            title_font_color='#60a5fa',
                            title_font_size=20,
                            legend=dict(
                                font=dict(color='white'),
                                bgcolor='rgba(30, 41, 59, 0.8)',
                                bordercolor='rgba(255, 255, 255, 0.2)',
                                borderwidth=1
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No economic data available for {country}")
                except Exception as e:
                    st.info(f"Economic indicator data not available: {e}")

            # Remove the education ranking bar chart and table
            # Only show QS Top Universities by Country in the third tab
            # (QS Top Universities table removed from here)
    except Exception as e:
        st.error(f"Error displaying life satisfaction metrics: {e}")

# ------------------------- SQLite Setup -------------------------
@st.cache_resource
def init_connection():
    try:
        conn = sqlite3.connect("predictions.db", check_same_thread=False)
        c = conn.cursor()
        
        # Create predictions table if it doesn't exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                age INTEGER,
                education TEXT,
                savings INTEGER,
                language_skills INTEGER,
                risk_tolerance INTEGER,
                career_field TEXT,
                predicted_country TEXT,
                risk_score REAL,
                opportunity_score REAL,
                lifestyle_score REAL
            )
        ''')
        
        # Create user feedback table if it doesn't exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country TEXT,
                feedback TEXT,
                rating INTEGER,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        return conn, c
    except sqlite3.Error as e:
        st.error(f"Database Error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error during database initialization: {e}")
        return None, None

# Initialize database connection
conn, c = None, None
try:
    conn, c = init_connection()
except Exception as e:
    st.error(f"Database connection error: {e}")

if not conn or not c:
    st.error("Failed to initialize database connection. Please check your database configuration.")
    st.stop()

# ------------------------- Custom CSS for Professional, Student-Friendly UI (Forceful) -------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*='css']  {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: #181f2a;
        color: #f4f4f9;
    }
    .main, .country-card, .pie-glass, .metric-container {
        background: #232a36;
        box-shadow: 0 2px 12px 0 rgba(24,31,42,0.10);
        border-radius: 16px;
        border: 1px solid #2d3748;
        margin-bottom: 1.5rem;
        padding: 1.2rem 1.2rem 1.2rem 1.2rem;
    }
    h1, h2, h3, h4 {
        color: #f4f4f9;
        font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f8cff 0%, #38bdf8 100%);
        color: #fff;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #0ea5e9 100%);
        color: #fff;
    }
    .metric-container {
        background: #232a36;
        color: #f4f4f9;
    }
    .stExpander, .stExpanderHeader {
        background: #232a36 !important;
        color: #f4f4f9 !important;
    }
    .education-table th {
        background: #232a36;
        color: #38bdf8;
    }
    .education-table tr:nth-child(even) {background: #232a36;}
    .education-table tr:nth-child(odd) {background: #1a202c;}
    .education-table td {color: #f4f4f9;}
    .stDataFrame, .stTable {
        background: #232a36 !important;
        color: #f4f4f9 !important;
    }
    .stMetric {
        background: #232a36 !important;
        color: #f4f4f9 !important;
    }
    .stSidebar {
        background: #1a202c !important;
        color: #f4f4f9 !important;
    }
    .stSelectbox, .stTextInput, .stSlider, .stNumberInput {
        background: #232a36 !important;
        color: #f4f4f9 !important;
        border-radius: 8px !important;
        border: 1px solid #2d3748 !important;
    }
    .stMarkdown a {
        color: #38bdf8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------- Title Section -------------------------
st.title("🌏 PathWise: Your Global Future Navigator")

st.markdown("""
    <div style='background: linear-gradient(135deg, #232946 0%, #334155 100%); 
                padding: 1.3rem; 
                border-radius: 16px; 
                margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(30,41,59,0.10);'>
        <h4 style='color: #a3e635; margin: 0; font-size: 1.4rem; letter-spacing: 1px; font-family:Inter,sans-serif;'>
            🎯 Make Informed Decisions About Your Global Future
        </h4>
        <p style='margin: 0.8rem 0 0 0; color: #cbd5e1; font-size: 1.08rem;'>
            Compare opportunities across multiple countries based on your profile, preferences, and goals.
        </p>
    </div>
""", unsafe_allow_html=True)

# ------------------------- Sidebar Inputs -------------------------
with st.sidebar:
    st.markdown("""
        <div style='background: linear-gradient(135deg, #334155 0%, #232946 100%);
                    padding: 1.1rem;
                    border-radius: 12px;
                    margin-bottom: 2rem;
                    color: #e0e7ef;
                    text-align: center;
                    box-shadow: 0 2px 12px rgba(30,41,59,0.10);'>
            <h3 style='margin: 0; color: #a3e635; letter-spacing: 1px; font-family:Inter,sans-serif;'>Your Profile</h3>
        </div>
    """, unsafe_allow_html=True)
    # Personal Information
    with st.expander("👤 Personal Details", expanded=True):
        age = st.slider("Age", 18, 60, 25,
            help="Your current age will help us tailor recommendations")
        education = st.selectbox("Education Level", 
            ["High School", "Bachelor's", "Master's", "PhD"],
            help="Your highest completed education level")
    career_field = st.selectbox(
        "Career Field",
        ["Technology", "Finance", "Healthcare", "Education", "Engineering", "Arts & Media", "Business", "Science & Research"],
        help="Select your primary career field"
    )

    # Skills and Resources
    with st.expander("💪 Skills & Resources", expanded=True):
        savings = st.slider("Savings (USD)", 0, 200000, 20000, step=5000, 
            format="$%d",
            help="Your current savings will affect opportunities and risk assessment")
        language_skills = st.slider("Language Skills", 1, 10, 5, 
            help="1 = Only Native Language, 10 = Multilingual Proficiency")
        risk_tolerance = st.slider("Risk Tolerance", 1, 10, 5,
            help="1 = Very Conservative, 10 = Very Risk-Taking")
        tuition_importance = st.slider("Tuition Affordability Importance", 1, 10, 7,
            help="1 = Cost is not a major factor, 10 = Seeking most affordable options")

    # Country Selection
    with st.expander("🌍 Countries to Compare", expanded=True):
        st.markdown("""
            <div style='margin-bottom: 1rem;'>
                <p style='color: #666; font-size: 0.9rem;'>
                    Select up to 5 countries to compare. Choose countries that align with your goals and interests.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        countries = st.multiselect(
            "Select Countries (Max 5)",
            ["USA", "Canada", "UK", "Australia", "Germany", "Japan", "Singapore", 
             "New Zealand", "Netherlands", "Sweden", "Switzerland", "Ireland"],
            default=["USA", "Canada", "UK"],
            max_selections=5
        )

# ------------------------- Analysis Button -------------------------
if st.sidebar.button("🔍 Analyze Opportunities", type="primary"):
    if not countries:
        st.warning("Please select at least one country to analyze.")
    else:
        with st.spinner("Analyzing global opportunities..."):
            time.sleep(1.2)
            try:
                # Create profile with validation
                profile = {
                    'age': max(18, min(60, age)),
                    'education': education,
                    'savings': max(0, min(200000, savings)),
                    'language_skills': max(1, min(10, language_skills)),
                    'risk_tolerance': max(1, min(10, risk_tolerance)),
                    'career_field': career_field,
                    'tuition_importance': max(1, min(10, tuition_importance))
                }
                
                # --- Analysis Logic ---
                results = []
                score_components_dict = {}
                
                for country in countries:
                    # Calculate scores
                    risk_score = calculate_risk_score(profile, country)
                    opportunity_score, score_components = calculate_opportunity_score(profile, country)
                    lifestyle_score = calculate_lifestyle_score(profile, country)
                    
                    # Store results
                    results.append({
                        "Country": country,
                        "Risk Score": risk_score,
                        "Opportunity Score": opportunity_score,
                        "Lifestyle Score": lifestyle_score
                    })
                    
                    score_components_dict[country] = score_components

                df_results = pd.DataFrame(results)
                st.success("✅ Analysis completed successfully!")

                st.header("🎯 Global Analysis Results")
                # --- Modern Card Layout for Country Analysis ---
                for idx, country_data in df_results.iterrows():
                    country = country_data['Country']
                    recommendations = get_country_recommendations(country, profile)
                    score_components = score_components_dict.get(country, {})

                    # --- Modern Card with Icon and Gradient ---
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #3847a3 0%, #232a36 100%);
                                    padding: 1.7rem 1.2rem 1.2rem 1.2rem;
                                    border-radius: 20px;
                                    margin: 1.5rem 0 1.5rem 0;
                                    border: 2px solid #a5b4fc;
                                    box-shadow: 0 8px 32px rgba(99,102,241,0.10); color:#f4f4f9;'>
                            <div style='display: flex; align-items: center;'>
                                <span style='font-size: 2.3rem; margin-right: 1.2rem;'>{'🏆' if country_data['Lifestyle Score'] == df_results['Lifestyle Score'].max() else '🌍'}</span>
                                <h2 style='color: #f4f4f9; margin: 0; font-size: 1.7rem; font-family:Inter,sans-serif;'>{country}</h2>
                            </div>
                            <div style='margin-top: 1.2rem; display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.2rem;'>
                                <div style='background: rgba(99,102,241,0.08); padding: 1rem; border-radius: 10px; color:#f4f4f9;'>
                                    <span style='color: #a5b4fc;'>Career Field</span>
                                    <div style='color: #f4f4f9; font-size: 1.13rem; font-weight:600;'>{profile['career_field']}</div>
                                </div>
                                <div style='background: rgba(99,102,241,0.08); padding: 1rem; border-radius: 10px; color:#f4f4f9;'>
                                    <span style='color: #a5b4fc;'>Education</span>
                                    <div style='color: #f4f4f9; font-size: 1.13rem; font-weight:600;'>{profile['education']}</div>
                                </div>
                                <div style='background: rgba(99,102,241,0.08); padding: 1rem; border-radius: 10px; color:#f4f4f9;'>
                                    <span style='color: #a5b4fc;'>Language Skills</span>
                                    <div style='color: #f4f4f9; font-size: 1.13rem; font-weight:600;'>{profile['language_skills']}/10</div>
                                </div>
                                <div style='background: rgba(99,102,241,0.08); padding: 1rem; border-radius: 10px; color:#f4f4f9;'>
                                    <span style='color: #a5b4fc;'>Financial Readiness</span>
                                    <div style='color: #f4f4f9; font-size: 1.13rem; font-weight:600;'>${profile['savings']:,}</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # --- Modern Metrics with Icons ---
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🏅 Overall Score", f"{country_data['Lifestyle Score']:.1f}%", 
                                  delta=f"{country_data['Lifestyle Score'] - df_results['Lifestyle Score'].mean():.1f}%")
                    with col2:
                        st.metric("⚠️ Risk Score", f"{country_data['Risk Score']:.1f}%")
                    with col3:
                        st.metric("🚀 Opportunity Score", f"{country_data['Opportunity Score']:.1f}%")

                    # --- Expanders with Modern Look ---
                    with st.expander("📊 View Detailed Analysis", expanded=False):
                        st.markdown("""
                            <div style='background: rgba(30, 41, 59, 0.7); padding: 1.2rem; border-radius: 10px; margin: 0.5rem 0;'>
                                <h4 style='color: #60a5fa; margin-bottom: 1rem;'>Score Components</h4>
                        """, unsafe_allow_html=True)
                        for label, key in [
                            ("Base Score", "base_score"),
                            ("Career Impact", "career_impact"),
                            ("Education Impact", "education_impact"),
                            ("Language Impact", "language_impact"),
                            ("Tuition Impact", "tuition_impact"),
                            ("Age Impact", "age_impact"),
                            ("Market Impact", "market_impact"),
                        ]:
                            st.markdown(
                                f"<div style='margin: 0.4rem 0; color: #93c5fd;'>{label}: <span style='color: white; float: right;'>{score_components.get(key, 0):.1f}</span></div>",
                                unsafe_allow_html=True
                            )
                        st.markdown("</div>", unsafe_allow_html=True)

                    with st.expander("🎯 Country-Specific Recommendations", expanded=True):
                        st.markdown(f"""
                            <div style='background: rgba(30, 41, 59, 0.7); padding: 1.2rem; border-radius: 10px;'>
                                <h4 style='color: #60a5fa; margin-bottom: 1rem;'>Key Advantages in {country}</h4>
                                <ul style='color: white; font-size: 1.05rem;'>
                                    <li>🔹 <strong>Tech & Innovation:</strong> {recommendations['Tech']}</li>
                                    <li>🎓 <strong>Education:</strong> {recommendations['Education']}</li>
                                    <li>🌟 <strong>Lifestyle:</strong> {recommendations['Lifestyle']}</li>
                                    <li>💡 <strong>Innovation:</strong> {recommendations['Innovation']}</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)

                    # --- Life Satisfaction Metrics ---
                    display_life_satisfaction_metrics(country)

                    # --- Language/Culture Buttons with Modern Look ---
                    if country in ["Japan", "Germany"]:
                        st.markdown("---")
                        lang_btn = st.button(f"🗣️ {country} Language Requirements", key=f"lang_req_{country}_{idx}")
                        cult_btn = st.button(f"{'🎌' if country == 'Japan' else '🇩🇪'} {country} Work Culture Guide", key=f"culture_guide_{country}_{idx}")
                        if lang_btn:
                            st.session_state[f"show_lang_req_{country}"] = True
                            st.session_state[f"show_culture_guide_{country}"] = False
                            st.experimental_rerun()
                        if cult_btn:
                            st.session_state[f"show_lang_req_{country}"] = False
                            st.session_state[f"show_culture_guide_{country}"] = True
                            st.experimental_rerun()
                        if st.session_state.get(f"show_lang_req_{country}", False):
                            st.markdown(f"""
                                <div style='background: rgba(30, 41, 59, 0.7);
                                            padding: 1.5rem;
                                            border-radius: 12px;
                                            margin-top: 1rem;'>
                                    <h3 style='color: #60a5fa; margin-bottom: 1rem;'>{country} Language Requirements</h3>
                            """, unsafe_allow_html=True)
                            display_language_requirements(country, profile)
                            st.markdown("</div>", unsafe_allow_html=True)
                        if st.session_state.get(f"show_culture_guide_{country}", False):
                            st.markdown(f"""
                                <div style='background: rgba(30, 41, 59, 0.7);
                                            padding: 1.5rem;
                                            border-radius: 12px;
                                            margin-top: 1rem;'>
                                    <h3 style='color: #60a5fa; margin-bottom: 1rem;'>{country} Work Culture Guide</h3>
                            """, unsafe_allow_html=True)
                            display_language_culture_tips(country, profile)
                            st.markdown("</div>", unsafe_allow_html=True)

                    # Define scores before using in visualizations
                    scores = {
                        'Risk': country_data['Risk Score'],
                        'Opportunity': country_data['Opportunity Score'],
                        'Lifestyle': country_data['Lifestyle Score']
                    }
                    # --- Modern Score Visualization ---
                    st.markdown(f"""
                        <div style='background: #1f2a3a;
                                    padding: 3rem; /* Increased padding */
                                    border-radius: 18px;
                                    margin: 2rem 0;
                                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                                    border: 1px solid #2d3e50;'>
                            <h3 style='color: #e0e7ff; text-align: center; margin-bottom: 2rem; font-size: 1.6rem; font-weight: 600;'>
                                {country} Performance Snapshot
                            </h3>
                    """, unsafe_allow_html=True)

                    # Create a modern gauge chart for overall score
                    overall_score = sum(scores.values())/len(scores)
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = overall_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'shape': 'angular',
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#7f8c8d"},
                            'bar': {'color': "#3498db", 'line': {'color': '#2980b9', 'width': 1}},
                            'bgcolor': "#2c3e50",
                            'borderwidth': 0,
                            'steps': [
                                # Update steps for Risk Score color mapping (Green, Yellow, Red) on Overall Score gauge
                                # These steps should now be correctly ordered and colored based on Risk Score ranges
                                {'range': [0, 30], 'color': '#2ecc71'}, # Green for Low Risk/Score
                                {'range': [30, 70], 'color': '#f39c12'}, # Yellow for Moderate Risk/Score
                                {'range': [70, 100], 'color': '#e74c3c'} # Red for High Risk/Score
                            ],
                            'threshold': {
                                'line': {'color': "#f8fafc", 'width': 4},
                                'thickness': 0.8,
                                'value': overall_score
                            }
                        },
                        number = {'font': {'color': "#e0e7ff", 'size': 32}}, # Further reduced number font size
                        title = {'text': "Overall Score", 'font': {'color': "#bdc3c7", 'size': 12}} # Further reduced title font size
                    ))
                    fig_gauge.update_layout(
                        height=140, # Further reduced height to make it much shorter
                        margin=dict(t=0, b=0, l=0, r=0), # Minimal margins
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#e0e7ff"}
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{country}")

                    # Create modern bar chart for individual metrics
                    metrics_df = pd.DataFrame({
                        'Metric': list(scores.keys()),
                        'Score': list(scores.values())
                    })
                    fig_metrics = go.Figure()
                    fig_metrics.add_trace(go.Bar(
                        x=metrics_df['Metric'],
                        y=metrics_df['Score'],
                        # Use distinct modern colors for each metric, slightly softer
                        marker=dict(
                            color=['#93c5fd', '#6ee7b7', '#fcd34d'], # Softer Blue, Green, Yellow
                            line=dict(color='#1f2a3a', width=0), # Remove border
                            cornerradius=3 # Slightly less rounded corners
                        ),
                        text=metrics_df['Score'].round(1),
                        textposition='outside',
                        textfont=dict(size=12, color='#e0e7ff'), # Slightly smaller text
                        hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>",
                        width=0.25 # Make bars very thin
                    ))
                    fig_metrics.update_layout(
                        title={
                            'text': "Detailed Metrics",
                            'y':0.95,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': {'color': '#bdc3c7', 'size': 16} # Slightly smaller title font
                        },
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#e0e7ff"},
                        bargap=0.5, # Significantly increase gap between bars
                        xaxis={
                            'showgrid': False,
                            'tickfont': {'color': '#bdc3c7', 'size': 11} # Slightly smaller tick font
                        },
                        yaxis={
                            'range': [0, 100],
                            'gridcolor': 'rgba(255, 255, 255, 0.05)', # Even less prominent grid lines
                            'tickfont': {'color': '#bdc3c7', 'size': 11}, # Slightly smaller tick font
                            'zeroline': False
                        },
                        height=260, # Slightly reduce height again
                        margin=dict(t=30, b=10, l=10, r=10) # Adjust margins
                    )
                    st.plotly_chart(fig_metrics, use_container_width=True, key=f"metrics_{country}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # --- Overall Country Comparison ---
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                                    padding: 2rem;
                                    border-radius: 16px;
                                    margin: 1.5rem 0;
                                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);'>
                            <h3 style='color: #f8fafc; text-align: center; margin-bottom: 1.5rem; font-size: 1.5rem;'>
                                Country Comparison Analysis
                            </h3>
                    """, unsafe_allow_html=True)

                    # Create modern radar chart for country comparison
                    categories = ['Risk', 'Opportunity', 'Lifestyle']
                    fig_radar = go.Figure()
                    
                    for _, country_data in df_results.iterrows():
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[country_data['Risk Score'], 
                               country_data['Opportunity Score'], 
                               country_data['Lifestyle Score']],
                            theta=categories,
                            fill='toself',
                            name=country_data['Country']
                        ))

                    # Generate a unique key for this analysis run (countries + timestamp)
                    countries_str = ','.join(df_results['Country'])
                    unique_hash = hashlib.md5((countries_str + str(time.time())).encode()).hexdigest()[:12]

                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100],
                                gridcolor='rgba(255, 255, 255, 0.1)',
                                tickfont={'color': '#f8fafc'}
                            ),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#f8fafc"},
                        showlegend=True,
                        legend=dict(
                            font={'color': '#f8fafc'},
                            bgcolor='rgba(30, 41, 59, 0.8)',
                            bordercolor='rgba(255, 255, 255, 0.2)',
                            borderwidth=1
                        ),
                        height=500,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, key=f"country_radar_comparison_{unique_hash}")

                    # Create modern bar chart for overall scores
                    overall_scores = []
                    for _, cd in df_results.iterrows():
                        overall_score = (
                            cd['Risk Score'] * 0.3 +
                            cd['Opportunity Score'] * 0.4 +
                            cd['Lifestyle Score'] * 0.3
                        )
                        overall_scores.append({'Country': cd['Country'], 'Overall Score': overall_score})

                    fig_comparison = go.Figure()
                    fig_comparison.add_trace(go.Bar(
                        x=[score['Country'] for score in overall_scores],
                        y=[score['Overall Score'] for score in overall_scores],
                        marker_color='#3b82f6',
                        text=[f"{score['Overall Score']:.1f}%" for score in overall_scores],
                        textposition='auto',
                        hovertemplate="<b>%{x}</b><br>Overall Score: %{y:.1f}%<extra></extra>",
                        width=0.3 # Make bars thinner
                    ))
                    fig_comparison.update_layout(
                        title={
                            'text': "Overall Country Scores",
                            'y':0.95,
                            'x':0.5,
                            'xanchor': 'center', # Center align title
                            'yanchor': 'top', # Align title to top
                            'font': {'color': '#f8fafc', 'size': 16} # Slightly smaller title font
                        },
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#f8fafc"},
                        bargap=0.5, # Increase gap between bars significantly
                        xaxis={
                            'gridcolor': 'rgba(255, 255, 255, 0.03)', # Even less prominent grid lines
                            'tickfont': {'color': '#f8fafc', 'size': 10} # Smaller tick font
                        },
                        yaxis={
                            'gridcolor': 'rgba(255, 255, 255, 0.03)', # Even less prominent grid lines
                            'tickfont': {'color': '#f8fafc', 'size': 10}, # Smaller tick font
                            'range': [0, 100]
                        },
                        height=280, # Reduce height
                        margin=dict(t=35, b=10, l=10, r=10) # Adjust margins
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True, key=f"country_bar_comparison_{unique_hash}")

                # --- Overall Comparison Pie Chart ---
                st.markdown(f'<div class="pie-glass"><h4 style="color:#6366f1; text-align:center; font-size:1.5rem; margin-bottom:0.5rem;">Overall Country Comparison</h4>', unsafe_allow_html=True)
                overall_scores = []
                for _, cd in df_results.iterrows():
                    overall_score = (
                        cd['Risk Score'] * 0.3 +
                        cd['Opportunity Score'] * 0.4 +
                        cd['Lifestyle Score'] * 0.3
                    )
                    overall_scores.append({'Country': cd['Country'], 'Overall Score': overall_score})
                fig_overall = go.Figure(data=[go.Pie(
                    labels=[score['Country'] for score in overall_scores],
                    values=[score['Overall Score'] for score in overall_scores],
                    hole=.5,  # Keep hole size
                    marker=dict(
                        colors=get_visualization_colors(len(overall_scores)),
                        line=dict(color='#1e293b', width=2)  # Darker border
                    ),
                    textinfo='label+percent',
                    textfont=dict(size=12, color='#f8fafc'),  # Further reduced text size
                    insidetextorientation='radial',
                    pull=[0.02]*len(overall_scores),  # Subtle pull effect
                    rotation=45,
                    hovertemplate="<b>%{label}</b><br>Overall Score: %{value:.1f}%<br>Percentage: %{percent}<br><extra></extra>",
                    sort=False
                )])
                fig_overall.update_layout(
                    title={
                        'text': "Country Score Distribution Comparison",
                        'y':0.98, # Adjust title position
                        'x':0.5,
                        'xanchor': 'center', # Center align title
                        'yanchor': 'top', # Align title to top
                        'font': {'color': '#f8fafc', 'size': 18}  # Keep title font size or slightly reduce if needed
                    },
                    paper_bgcolor='rgba(30, 41, 59, 0.8)',
                    plot_bgcolor='rgba(30, 41, 59, 0.8)',
                    height=300,  # Significantly reduced height
                    showlegend=False,
                    margin=dict(t=50, b=10, l=10, r=10),  # Adjusted margins
                    annotations=[dict(
                        text=f"Average<br>{sum(s['Overall Score'] for s in overall_scores)/len(overall_scores):.1f}%",
                        x=0.5,
                        y=0.5,
                        font_size=16, # Further reduced annotation font size
                        showarrow=False,
                        font_color='#f8fafc'
                    )]
                )
                st.plotly_chart(fig_overall, use_container_width=True, key=f"overall_comparison_{unique_hash}")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.stop()

# ------------------------- History Section -------------------------
st.markdown("---")
st.markdown("## 🕑 Analysis History")
with st.expander("📊 View Analysis History", expanded=False):
    try:
        c.execute("""
            SELECT timestamp, age, education, career_field, 
                   predicted_country, risk_score, opportunity_score, lifestyle_score 
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        rows = c.fetchall()
        if rows:
            df_history = pd.DataFrame(rows, columns=[
                "Timestamp", "Age", "Education", "Career Field",
                "Country", "Risk Score", "Opportunity Score", "Lifestyle Score"
            ])
            st.dataframe(df_history, use_container_width=True)
        else:
            st.info("No previous analysis available.")
    except sqlite3.Error as e:
        st.error(f"Database Error: {e}")

# ------------------------- Upload Additional Data Section -------------------------
st.markdown("---")
st.markdown("## 📁 Upload Additional Data (CSV or Other Files)")
with st.expander("➕ Upload File", expanded=False):
    uploaded_file = st.file_uploader("Choose a CSV or data file to upload", type=["csv", "xlsx", "txt", "json"])
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        if uploaded_file.name.endswith('.csv'):
            df_uploaded = pd.read_csv(uploaded_file)
            st.dataframe(df_uploaded.head(), use_container_width=True)
        elif uploaded_file.name.endswith('.xlsx'):
            df_uploaded = pd.read_excel(uploaded_file)
            st.dataframe(df_uploaded.head(), use_container_width=True)
        elif uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            st.json(data)
        elif uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8')
            st.text(content[:1000])
        else:
            st.info("File type not previewable, but upload succeeded.")

# ------------------------- Global Country Rankings Section -------------------------
st.markdown("---")
st.markdown("## 🌐 Global Country Rankings")

# Create tabs for different ranking categories
ranking_tabs = st.tabs(["📊 HDI Rankings", "💰 GDP Rankings", "📋 Detailed Education List by Country"])

with ranking_tabs[0]:
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    border: 1px solid #475569;'>
            <h3 style='color: #60a5fa; 
                       margin-bottom: 1rem;
                       font-size: 1.5rem;'>
                📊 Human Development Index (HDI) Rankings
            </h3>
            <p style='color: #93c5fd; font-size: 1rem;'>
                The Human Development Index (HDI) is a composite statistic of life expectancy, education, and per capita income indicators.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        hdi_df = pd.read_csv('hdi_ranking.csv')
        fig_hdi = px.bar(hdi_df.sort_values('Rank'), 
                        x='Country', 
                        y='HDI', 
                        title='UN HDI by Country', 
                        color='HDI', 
                        color_continuous_scale='Blues')
        fig_hdi.update_layout(
            plot_bgcolor='rgba(30, 41, 59, 0.3)',
            paper_bgcolor='rgba(30, 41, 59, 0.3)',
            font_color='white',
            title_font_color='#60a5fa',
            title_font_size=20,
            showlegend=False
        )
        st.plotly_chart(fig_hdi, use_container_width=True)
    except Exception as e:
        st.info("HDI ranking data not available. Please check the data source.")

with ranking_tabs[1]:
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    border: 1px solid #475569;'>
            <h3 style='color: #60a5fa; 
                       margin-bottom: 1rem;
                       font-size: 1.5rem;'>
                💰 GDP Rankings
            </h3>
            <p style='color: #93c5fd; font-size: 1rem;'>
                Gross Domestic Product (GDP) rankings show the economic strength and size of different countries.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        gdp_df = pd.read_csv('gdp_ranking.csv')
        fig_gdp = px.bar(gdp_df.sort_values('Rank'), 
                        x='Country', 
                        y='GDP', 
                        title='GDP by Country (USD)', 
                        color='GDP', 
                        color_continuous_scale='Greens')
        fig_gdp.update_layout(
            plot_bgcolor='rgba(30, 41, 59, 0.3)',
            paper_bgcolor='rgba(30, 41, 59, 0.3)',
            font_color='white',
            title_font_color='#60a5fa',
            title_font_size=20,
            showlegend=False
        )
        st.plotly_chart(fig_gdp, use_container_width=True)
    except Exception as e:
        st.info("GDP ranking data not available. Please check the data source.")

with ranking_tabs[2]:
    st.markdown('''
        <div style='background: linear-gradient(135deg, #6366f1 0%, #0ea5e9 100%); padding: 1.5rem; border-radius: 12px; border: 1px solid #60a5fa; margin-bottom: 1.5rem;'>
            <h3 style='color: #fff; margin-bottom: 1rem; font-size: 1.7rem; letter-spacing: 1px;'>📋 Detailed Education List by Country</h3>
            <p style='color: #e0e7ff; font-size: 1.08rem;'>Compare <b>top universities</b>, <span style="color:#fbbf24;">tuition costs</span>, and <span style="color:#34d399;">language requirements</span> for each country. Hover for details.</p>
        </div>
    ''', unsafe_allow_html=True)
    qs_universities = [
        {'Country': 'USA', 'Top Universities': 'Massachusetts Institute of Technology, Stanford University, Harvard University, California Institute of Technology, University of Chicago', 'Avg Tuition (USD)': 55000, 'Language': 'English'},
        {'Country': 'UK', 'Top Universities': 'University of Oxford, University of Cambridge, Imperial College London, University College London, University of Edinburgh', 'Avg Tuition (USD)': 34000, 'Language': 'English'},
        {'Country': 'Germany', 'Top Universities': 'Technical University of Munich, Heidelberg University, Ludwig Maximilian University of Munich, Humboldt University of Berlin, RWTH Aachen University', 'Avg Tuition (USD)': 300, 'Language': 'German'},
        {'Country': 'Japan', 'Top Universities': 'University of Tokyo, Kyoto University, Osaka University, Tokyo Institute of Technology, Tohoku University', 'Avg Tuition (USD)': 5350, 'Language': 'Japanese'},
        {'Country': 'Canada', 'Top Universities': 'University of Toronto, University of British Columbia, McGill University, University of Montreal, University of Alberta', 'Avg Tuition (USD)': 39000, 'Language': 'English'},
        {'Country': 'Australia', 'Top Universities': 'University of Melbourne, University of Sydney, Australian National University, University of Queensland, University of New South Wales', 'Avg Tuition (USD)': 41000, 'Language': 'English'},
        {'Country': 'Singapore', 'Top Universities': 'National University of Singapore, Nanyang Technological University', 'Avg Tuition (USD)': 34000, 'Language': 'English'}
    ]
    import pandas as pd
    import plotly.express as px
    edu_df = pd.DataFrame(qs_universities)
    # Modern styled table with color highlights for top universities
    def highlight_universities(val):
        unis = val.split(', ')
        return ', '.join([f'<span style="color:#fbbf24;font-weight:bold;">{u}</span>' if i == 0 else u for i, u in enumerate(unis)])
    edu_df['Top Universities'] = edu_df['Top Universities'].apply(highlight_universities)
    st.markdown('''<style>.education-table td, .education-table th {padding: 12px 18px; font-size: 1.08rem;} .education-table th {background: #1e293b; color: #60a5fa;} .education-table tr:nth-child(even) {background: #f8fafc;} .education-table tr:nth-child(odd) {background: #ffffff;} .education-table td {color: #334155;} .education-table {border-radius: 10px; overflow: hidden; box-shadow: 0 4px 16px rgba(59,130,246,0.10);}</style>''', unsafe_allow_html=True)
    st.markdown(edu_df.to_html(escape=False, index=False, justify='center', classes='education-table'), unsafe_allow_html=True)
    # Tuition cost bar chart
    fig_tuition = px.bar(edu_df, x='Country', y='Avg Tuition (USD)', color='Avg Tuition (USD)', color_continuous_scale='Blues',
        title='Average Tuition Cost by Country', labels={'Avg Tuition (USD)': 'Avg Tuition (USD)'})
    fig_tuition.update_layout(
        plot_bgcolor='rgba(30, 41, 59, 0.3)',
        paper_bgcolor='rgba(30, 41, 59, 0.3)',
        font_color='white',
        title_font_color='#60a5fa',
        title_font_size=20,
        showlegend=False
    )
    st.plotly_chart(fig_tuition, use_container_width=True)
    # German/Japan language requirement and study prep section
    st.markdown('''
<div style='margin-top:2.5rem; margin-bottom:2.5rem; display:flex; gap:2.5rem; flex-wrap:wrap;'>
    <div style='flex:1; min-width:320px; background:linear-gradient(135deg,#e0e7ff 0%,#bae6fd 100%); border-radius:14px; padding:1.5rem; box-shadow:0 2px 12px rgba(59,130,246,0.10);'>
        <h4 style='color:#2563eb; margin-bottom:0.7rem;'>🇩🇪 German Language Study Prep</h4>
        <ul style='color:#232946; font-size:1.08rem; line-height:1.7;'>
            <li><b>Minimum Level:</b> B2 (for most programs), C1 for advanced/academic</li>
            <li><b>Recommended Test:</b> TestDaF or DSH</li>
            <li><b>Prep Tips:</b> Take integration courses, use Goethe-Institut resources, join language tandems, practice academic writing</li>
            <li><b>Study Duration:</b> 6-12 months for B2/C1 if starting from A2</li>
            <li><b>Free/Low Tuition:</b> Most public universities</li>
        </ul>
    </div>
    <div style='flex:1; min-width:320px; background:linear-gradient(135deg,#fef9c3 0%,#fbbf24 100%); border-radius:14px; padding:1.5rem; box-shadow:0 2px 12px rgba(251,191,36,0.10);'>
        <h4 style='color:#b45309; margin-bottom:0.7rem;'>🇯🇵 Japanese Language Study Prep</h4>
        <ul style='color:#232946; font-size:1.08rem; line-height:1.7;'>
            <li><b>Minimum Level:</b> JLPT N2 (for most programs), N1 for top/academic</li>
            <li><b>Recommended Test:</b> JLPT</li>
            <li><b>Prep Tips:</b> Take intensive language courses, use Japan Foundation resources, join language exchange, focus on kanji and academic vocabulary</li>
            <li><b>Study Duration:</b> 9-18 months for N2/N1 if starting from N4</li>
            <li><b>Affordable Tuition:</b> Most national universities</li>
        </ul>
    </div>
</div>
''', unsafe_allow_html=True)

# ------------------------- Analytics Section -------------------------
st.markdown("---")
st.markdown("""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                padding: 2rem;
                border-radius: 12px;
                margin: 2rem 0;
                border: 1px solid #60a5fa;'>
        <h2 style='color: #60a5fa; 
                   font-size: 2rem; 
                   margin-bottom: 1rem;
                   text-align: center;'>
            📊 PathWise Analytics
        </h2>
        <p style='color: #93c5fd; 
                  text-align: center; 
                  font-size: 1.1rem;
                  margin-bottom: 2rem;'>
            Discover insights from global user experiences and decisions
        </p>
    </div>
""", unsafe_allow_html=True)

# Create tabs for different analytics sections
analytics_tabs = st.tabs(["🌍 Popular Destinations", "👥 Career Field Trends", "📈 Country Score Analysis"])

with analytics_tabs[0]:
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    border: 1px solid #475569;'>
            <h3 style='color: #60a5fa; 
                       margin-bottom: 1rem;
                       font-size: 1.5rem;'>
                🌎 Popular Destinations
            </h3>
            <p style='color: #93c5fd; font-size: 1rem;'>
                Most frequently selected countries by users based on their profiles and preferences.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    popular_countries = get_popular_countries()
    if popular_countries:
        df_popular = pd.DataFrame(popular_countries, columns=['Country', 'Count'])
        fig_popular = px.bar(
            df_popular,
            x='Country',
            y='Count',
            title='Most Popular Country Choices',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig_popular.update_layout(
            plot_bgcolor='rgba(30, 41, 59, 0.3)',
            paper_bgcolor='rgba(30, 41, 59, 0.3)',
            font_color='white',
            title_font_color='#60a5fa',
            title_font_size=20,
            showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_popular, use_container_width=True)
    else:
        st.info("No data available for popular destinations yet.")

with analytics_tabs[1]:
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    border: 1px solid #475569;'>
            <h3 style='color: #60a5fa; 
                       margin-bottom: 1rem;
                       font-size: 1.5rem;'>
                👥 Career Field Trends
            </h3>
            <p style='color: #93c5fd; font-size: 1rem;'>
                Distribution of career fields across different countries based on user selections.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    career_trends = get_career_trends()
    if career_trends:
        df_career = pd.DataFrame(career_trends, columns=['Career Field', 'Country', 'Count'])
        fig_career = px.treemap(
            df_career,
            path=['Career Field', 'Country'],
            values='Count',
            title='Career Fields by Country',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig_career.update_layout(
            plot_bgcolor='rgba(30, 41, 59, 0.3)',
            paper_bgcolor='rgba(30, 41, 59, 0.3)',
            font_color='white',
            title_font_color='#60a5fa',
            title_font_size=20,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_career, use_container_width=True)
    else:
        st.info("No data available for career field trends yet.")

with analytics_tabs[2]:
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    border: 1px solid #475569;'>
            <h3 style='color: #60a5fa; 
                       margin-bottom: 1rem;
                       font-size: 1.5rem;'>
                📈 Country Score Analysis
            </h3>
            <p style='color: #93c5fd; font-size: 1rem;'>
                Detailed analysis of risk, opportunity, and lifestyle scores across different countries.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    avg_scores = get_average_scores()
    if avg_scores:
        df_scores = pd.DataFrame(avg_scores, columns=['Country', 'Risk', 'Opportunity', 'Lifestyle'])
        
        # Create comparison pie chart
        fig_compare = go.Figure()
        
        for metric in ['Risk', 'Opportunity', 'Lifestyle']:
            fig_compare.add_trace(go.Pie(
                labels=df_scores['Country'],
                values=df_scores[metric],
                name=metric,
                hole=.3,
                marker=dict(
                    colors=get_visualization_colors(len(df_scores)),
                    line=dict(color='#ffffff', width=2)
                ),
                textinfo='label+percent',
                textfont=dict(size=15, color='white'),
                hovertemplate="<b>%{label}</b><br>" +
                            f"{metric}: %{{y:.1f}}%<br>" +
                            "Percentage: %{percent}<br>" +
                            "<extra></extra>"
            ))
        
        fig_compare.update_layout(
            title={
                'text': "Country Score Distribution Comparison",
                'y':0.95,
                'x':0.5,
                'font': {'color': 'white', 'size': 20}
            },
            paper_bgcolor='rgba(30, 41, 59, 0.8)',
            plot_bgcolor='rgba(30, 41, 59, 0.8)',
            height=500,
            showlegend=True,
            legend=dict(
                font={'color': 'white'},
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor='rgba(255, 255, 255, 0.2)',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Add bar chart for detailed comparison
        fig_bar = go.Figure()
        colors = get_visualization_colors(3)
        
        for idx, metric in enumerate(['Risk', 'Opportunity', 'Lifestyle']):
            fig_bar.add_trace(go.Bar(
                name=metric,
                x=df_scores['Country'],
                y=df_scores[metric],
                text=df_scores[metric].round(1),
                textposition='auto',
                marker_color=colors[idx],
                hovertemplate="<b>%{x}</b><br>" +
                            f"{metric}: %{{y:.1f}}%<br>" +
                            "<extra></extra>"
            ))
        
        fig_bar.update_layout(
            title={
                'text': "Detailed Score Comparison",
                'font': {'color': 'white', 'size': 20},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            barmode='group',
            height=400,
            paper_bgcolor='rgba(30, 41, 59, 0.8)',
            plot_bgcolor='rgba(30, 41, 59, 0.8)',
            font={'color': 'white'},
            xaxis={
                'gridcolor': 'rgba(255, 255, 255, 0.1)',
                'tickfont': {'color': 'white'},
                'title': {'text': 'Countries', 'font': {'color': 'white'}}
            },
            yaxis={
                'gridcolor': 'rgba(255, 255, 255, 0.1)',
                'tickfont': {'color': 'white'},
                'title': {'text': 'Score (%)', 'font': {'color': 'white'}},
                'tickformat': '.1%'
            },
            legend={
                'font': {'color': 'white'},
                'bgcolor': 'rgba(30, 41, 59, 0.8)',
                'bordercolor': 'rgba(255, 255, 255, 0.2)',
                'borderwidth': 1
            },
            hoverlabel={'font': {'size': 14}}
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No data available for country score analysis yet.")

class LifeOutcome:
    def __init__(self, happiness, income, stress):
        self.happiness = happiness
        self.income = income
        self.stress = stress

def calculate_weighted_score(outcome):
    return (2 * outcome.happiness + 2 * outcome.income - outcome.stress) / 5

# ------------------------- User Feedback Section -------------------------
st.markdown("---")
st.markdown("## 📝 User Feedback & Ratings")
with st.form("feedback_form"):
    feedback_rating = st.slider("How useful were the recommendations?", 1, 5, 3)
    feedback_comment = st.text_area("Any comments or suggestions?")
    submitted = st.form_submit_button("Submit Feedback")
    if submitted:
        try:
            c.execute("INSERT INTO user_feedback (country, feedback, rating, timestamp) VALUES (?, ?, ?, ?)",
                      (', '.join(countries), feedback_comment, feedback_rating, datetime.datetime.now().isoformat()))
            conn.commit()
            st.success("Thank you for your feedback!")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")

# ------------------------- Download/Export Results Section -------------------------
st.markdown("---")
st.markdown("## 📥 Download Your Analysis Results")
if 'df_results' in locals():
    st.download_button(
        label="Download Results as CSV",
        data=df_results.to_csv(index=False),
        file_name="pathwise_analysis_results.csv",
        mime="text/csv"
    )

# ------------------------- Personalized Recommendations Breakdown -------------------------
if 'df_results' in locals():
    st.markdown("---")
    st.markdown("## 🧭 Why This Country? (Top Factors)")
    st.markdown("""
    **What do these scores mean?**
    - **Opportunity Score:** How many opportunities (jobs, education) match your profile. Higher is better.
    - **Lifestyle Score:** Expected quality of life, including social, cultural, and well-being factors. Higher is better.

    **⚠️ What Does Risk Score Mean?**
    ➤ Risk Score = How risky a life path is

    | Risk Score (%) | Meaning           | Color       |
    |----------------|-------------------|-------------|
    | 0–30%          | Low risk (good)   | 🟢 Green    |
    | 30–70%         | Moderate risk     | 🟡 Yellow-ish |
    | 70–100%        | High risk         | 🔴 Red      |
    """, unsafe_allow_html=True)
    for idx, country_data in df_results.iterrows():
        country = country_data['Country']
        st.markdown(f"**{country}:** ")
        st.markdown(f"- Risk Score: {country_data['Risk Score']:.1f}%")
        st.markdown(f"- Opportunity Score: {country_data['Opportunity Score']:.1f}%")
        st.markdown(f"- Lifestyle Score: {country_data['Lifestyle Score']:.1f}%")

# ------------------------- Save/Load User Profile Section -------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💾 Save/Load Profile")
    if st.button("Save Profile"):
        st.session_state['saved_profile'] = {
            'age': age,
            'education': education,
            'savings': savings,
            'language_skills': language_skills,
            'risk_tolerance': risk_tolerance,
            'career_field': career_field,
            'tuition_importance': tuition_importance
        }
        st.success("Profile saved!")
    if st.button("Load Profile") and 'saved_profile' in st.session_state:
        saved = st.session_state['saved_profile']
        age = saved['age']
        education = saved['education']
        savings = saved['savings']
        language_skills = saved['language_skills']
        risk_tolerance = saved['risk_tolerance']
        career_field = saved['career_field']
        tuition_importance = saved['tuition_importance']
        st.success("Profile loaded!")

# ------------------------- Language Switcher -------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🌐 Language")
    st.selectbox("Select Language", ["English"], index=0, help="More languages coming soon!")

# ------------------------- Data Sources Info Button -------------------------
st.markdown("---")
st.markdown("## 📚 Data Sources")
with st.expander("Show Data Sources"):
    st.markdown("""
    - [UN Human Development Index](https://hdr.undp.org/data-center/human-development-index)
    - [World Bank GDP Data](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD)
    - [QS University Rankings](https://www.topuniversities.com/university-rankings)
    - Custom CSVs for tuition, language, and life satisfaction
    """)

# ------------------------- How to Use / FAQ Section -------------------------
st.markdown("---")
st.markdown("## ❓ How to Use / FAQ")
with st.expander("Show FAQ"):
    st.markdown("""
    - **How do I use PathWise?**
      - Fill out your profile in the sidebar, select countries, and click Analyze Opportunities.
    - **What do the scores mean?**
      - **Risk Score:** How safe and stable the country is for you. Higher is better. A high score means fewer barriers or uncertainties for your profile.
      - **Opportunity Score:** How many opportunities (jobs, education) match your profile. Higher is better. A low score means there may be limited opportunities or more competition.
      - **Lifestyle Score:** Expected quality of life, including social, cultural, and well-being factors. Higher is better. A high score means a better quality of life and environment.
    - **Can I upload my own data?**
      - Yes! Use the Upload Additional Data section below.
    - **How do I save my profile?**
      - Use the Save/Load Profile buttons in the sidebar.
    """)

# ------------------------- Responsive CSS Tweaks for Mobile -------------------------
st.markdown("""
    <style>
    @media (max-width: 600px) {
        .main, .country-card, .pie-glass, .metric-container {
            padding: 0.5rem !important;
        }
        h1, h2, h3, h4 {
            font-size: 1.1rem !important;
        }
        .stButton>button {
            font-size: 1rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------- Progress Bar / Stepper -------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🚦 Progress")
    progress = 0
    if age: progress += 1
    if education: progress += 1
    if savings: progress += 1
    if language_skills: progress += 1
    if risk_tolerance: progress += 1
    if career_field: progress += 1
    if tuition_importance: progress += 1
    st.progress(progress/7, text=f"{progress}/7 profile steps completed")
