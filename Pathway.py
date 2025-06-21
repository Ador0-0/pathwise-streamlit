# Standard library imports
import sqlite3
import base64
from io import BytesIO
import requests
import json
import logging

# Third-party imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ------------------------- Configuration and Setup -------------------------

# Page configuration
st.set_page_config(
    page_title="PathWise",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_visualization_colors(n_colors):
    """Generate a list of distinct colors for visualizations."""
    # Using a predefined color scale from Plotly
    colors = px.colors.qualitative.Plotly
    return colors * (n_colors // len(colors)) + colors[:n_colors % len(colors)]

# ------------------------- Core Scoring Functions -------------------------

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
    """Fetch and display career trends"""
    try:
        # This is a placeholder; in a real app, this would come from an API or database
        trends = {
            "Technology": ["AI/ML Specialist", "Data Scientist", "Cybersecurity Analyst"],
            "Healthcare": ["Telemedicine Physician", "Nurse Practitioner", "Health Informatics Specialist"],
            "Finance": ["Fintech Analyst", "Robo-Advisor", "Sustainable Finance Expert"],
            "Education": ["Online Curriculum Developer", "Instructional Designer", "EdTech Specialist"]
        }
        return trends
    except Exception:
        return {}

def get_average_scores():
    """Calculate average scores from the database"""
    try:
        c.execute("""
            SELECT 
                AVG(risk_score) as avg_risk,
                AVG(opportunity_score) as avg_opp,
                AVG(lifestyle_score) as avg_life
            FROM predictions
        """)
        return c.fetchone()
    except sqlite3.Error:
        return (50, 50, 50)

# ------------------------- Data Loading and Processing -------------------------

def load_external_data():
    """Load all necessary CSV files into dataframes."""
    try:
        hdi_df = pd.read_csv('hdi_ranking.csv')
        gdp_df = pd.read_csv('gdp_ranking.csv')
        uni_data = pd.read_csv('university_rankings.csv')
        tuition_df = pd.read_csv('tuition_part_time.csv')
        gdp_growth_df = pd.read_csv('gdp_growth.csv')
        job_market_df = pd.read_csv('job_market_data.csv')
        life_satisfaction_df = pd.read_csv('life_satisfaction_metrics.csv')
        language_df = pd.read_csv('language_difficulty.csv')
        
        # Clean data - for example, remove '%' and convert to float
        if 'GDP_Growth' in gdp_growth_df.columns:
            gdp_growth_df['GDP_Growth'] = gdp_growth_df['GDP_Growth'].str.replace('%', '').astype(float)
            
        return hdi_df, gdp_df, uni_data, tuition_df, gdp_growth_df, job_market_df, life_satisfaction_df, language_df
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e.filename}. Please ensure all CSV files are in the repository.")
        return [pd.DataFrame() for _ in range(8)]
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        return [pd.DataFrame() for _ in range(8)]

def process_uploaded_csv(uploaded_file):
    """Process an uploaded CSV file for batch analysis"""
    try:
        df = pd.read_csv(uploaded_file)
        results = []
        
        # Check for required columns
        required_cols = ['age', 'education', 'savings', 'language_skills', 'risk_tolerance', 'career_field', 'tuition_importance', 'countries']
        if not all(col in df.columns for col in required_cols):
            st.error("Uploaded CSV is missing required columns.")
            return None
            
        for index, row in df.iterrows():
            profile = {
                'age': row['age'],
                'education': row['education'],
                'savings': row['savings'],
                'language_skills': row['language_skills'],
                'risk_tolerance': row['risk_tolerance'],
                'career_field': row['career_field'],
                'tuition_importance': row['tuition_importance']
            }
            countries = [c.strip() for c in row['countries'].split(',')]
            
            for country in countries:
                risk_score = calculate_risk_score(profile, country)
                opportunity_score = calculate_opportunity_score(profile, country)
                lifestyle_score = calculate_lifestyle_score(profile, country)
                results.append([row.get('id', index), country, risk_score, opportunity_score, lifestyle_score])
                
        result_df = pd.DataFrame(results, columns=['ID', 'Country', 'Risk Score', 'Opportunity Score', 'Lifestyle Score'])
        return result_df
        
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return None

def analyze_part_time_work(country):
    """Analyze part-time work opportunities for a given country"""
    try:
        df = pd.read_csv('part_time_work_for_students.csv')
        country_data = df[df['Country'] == country]
        
        if not country_data.empty:
            data = country_data.iloc[0]
            return {
                "hours_allowed": data.get('Hours_Allowed', 'N/A'),
                "visa_type": data.get('Visa_Type', 'N/A'),
                "minimum_wage_local": data.get('Minimum_Wage_Local', 'N/A'),
                "notes": data.get('Notes', 'No specific notes available.')
            }
        else:
            return None
    except FileNotFoundError:
        st.warning(f"Part-time work data for {country} not found.")
        return None
    except Exception as e:
        st.error(f"Error analyzing part-time work: {e}")
        return None

def analyze_language_requirements(country, profile):
    """Analyze language requirements based on country and user profile"""
    try:
        df = pd.read_csv('language_difficulty.csv')
        country_data = df[df['Country'] == country]
        
        if not country_data.empty:
            data = country_data.iloc[0]
            language = data.get('Official_Language', 'N/A')
            difficulty = data.get('Difficulty_Score', 3)  # Default to moderate
            
            # Simple logic: higher skill level reduces the barrier
            language_barrier = max(0, difficulty * 10 - profile.get('language_skills', 5) * 1.5)
            
            return {
                "language": language,
                "difficulty_score": difficulty,
                "estimated_barrier": language_barrier, # A score from 0-30
                "notes": data.get('Learning_Tips', 'Focus on conversational practice.')
            }
        else:
            return None
    except FileNotFoundError:
        st.warning(f"Language requirement data for {country} not found.")
        return None
    except Exception as e:
        st.error(f"Error analyzing language requirements: {e}")
        return None

# Initialize logging
LOG_FILE = "app_activity.log"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])

# Function to fetch country flag from an API
@st.cache_data
def get_country_flag(country_name):
    """Fetch country flag from an API by name."""
    try:
        # A more robust mapping for country names to country codes
        country_mapping = {
            "United States": "USA",
            "United Kingdom": "GBR",
            "UAE": "ARE"
            # Add other mappings as needed
        }
        # Fallback to the country name if not in mapping
        country_code = country_mapping.get(country_name, country_name)
        
        response = requests.get(f"https://restcountries.com/v3.1/name/{country_code}")
        response.raise_for_status() # Raise an exception for bad status codes
        
        # The API can return multiple results; we take the first one.
        data = response.json()
        if data:
            return data[0]['flags']['png']
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for {country_name}: {e}")
    except (KeyError, IndexError):
        logging.error(f"Could not parse flag for {country_name} from API response.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_country_flag for {country_name}: {e}")
    
    # Return a default placeholder if anything fails
    return "https://via.placeholder.com/40x30.png?text=?"


# Function to get country info from an API
@st.cache_data
def get_country_info(country_name):
    """Fetch detailed country information from an API."""
    try:
        response = requests.get(f"https://restcountries.com/v3.1/name/{country_name}?fullText=true")
        response.raise_for_status()
        data = response.json()
        if data:
            info = data[0]
            return {
                "capital": info.get('capital', ['N/A'])[0],
                "population": f"{info.get('population', 0):,}",
                "region": info.get('region', 'N/A'),
                "subregion": info.get('subregion', 'N/A'),
                "languages": ", ".join(info.get('languages', {}).values()),
                "currencies": ", ".join([c['name'] for c in info.get('currencies', {}).values()])
            }
    except Exception as e:
        logging.error(f"Could not fetch country info for {country_name}: {e}")
    return None

def display_country_dashboard(country):
    """Display a dashboard of information for a selected country."""
    info = get_country_info(country)
    if info:
        st.subheader(f"🌍 About {country}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Capital", info['capital'])
        col2.metric("Population", info['population'])
        col3.metric("Region", info['region'])
        
        with st.expander("More Details"):
            st.write(f"**Subregion:** {info['subregion']}")
            st.write(f"**Languages:** {info['languages']}")
            st.write(f"**Currencies:** {info['currencies']}")
    else:
        st.info(f"Could not load detailed information for {country}.")


# Function to generate PDF report
def generate_pdf_report(profile, results_df, recommendation):
    """Generate a downloadable PDF report of the analysis."""
    try:
        from fpdf import FPDF
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'PathWise Analysis Report', 0, 1, 'C')

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        pdf = PDF()
        pdf.add_page()
        
        # Profile Section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Your Profile', 0, 1)
        pdf.set_font('Arial', '', 10)
        for key, value in profile.items():
            pdf.cell(0, 7, f"{key.replace('_', ' ').title()}: {value}", 0, 1)
        
        # Recommendation Section
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Top Recommendation', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 7, recommendation)

        # Full Results Table
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Full Results', 0, 1)
        pdf.set_font('Arial', 'B', 10)
        
        # Table Header
        col_widths = [40, 35, 35, 35]
        for i, header in enumerate(results_df.columns):
            pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
        pdf.ln()

        # Table Body
        pdf.set_font('Arial', '', 10)
        for index, row in results_df.iterrows():
            for i, item in enumerate(row):
                 pdf.cell(col_widths[i], 10, str(round(item, 2) if isinstance(item, float) else item), 1, 0, 'C')
            pdf.ln()
            
        return pdf.output(dest='S').encode('latin1')
    
    except ImportError:
        st.error("FPDF library not found. Please install it to generate PDFs.")
        return None
    except Exception as e:
        st.error(f"Failed to generate PDF report: {e}")
        return None


# ------------------------- UI Display Functions -------------------------

def display_language_culture_tips(country, profile):
    """Display tips related to language and culture for a specific country."""
    
    st.info(f"#### Cultural & Language Insights for {country}")

    # Language Analysis
    lang_analysis = analyze_language_requirements(country, profile)
    if lang_analysis:
        st.write(f"**Official Language:** {lang_analysis['language']}")
        
        # Progress bar for language barrier
        barrier_score = lang_analysis['estimated_barrier']
        st.progress(barrier_score / 30, text=f"Estimated Language Barrier: {int(barrier_score)}/30")
        
        if barrier_score > 20:
            st.warning("High language barrier detected. Intensive study may be required.")
        elif barrier_score > 10:
            st.info("Moderate language barrier. Consistent effort in learning is advisable.")
        else:
            st.success("Low language barrier. Your existing skills give you a great start!")
        
        st.write(f"**Learning Tip:** *{lang_analysis['notes']}*")

    # Cultural Tips (Placeholder - could be from a CSV or API)
    cultural_tips = {
        "Japan": "Bowing is a sign of respect. Punctuality is highly valued. Avoid tipping.",
        "Germany": "Be direct and punctual. Shake hands firmly. Respect 'quiet hours' (Ruhezeit).",
        "Canada": "Politeness and apologizing are common. Embrace multiculturalism. Tipping is standard.",
        "United States": "Individualism is valued. Small talk is common. Tipping is expected in service industries.",
        "United Kingdom": "Queuing is a must. 'Please' and 'thank you' are essential. Pub culture is central.",
        "Australia": "Laid-back and informal. Use 'mate' widely. Barbecues ('barbies') are a social staple."
    }
    
    if country in cultural_tips:
        st.write(f"**Cultural Tip:** *{cultural_tips[country]}*")

def display_language_requirements(country, profile):
    """Displays detailed language requirements and tips."""
    
    language_analysis = analyze_language_requirements(country, profile)
    
    if language_analysis:
        st.subheader(f"🗣️ Language Deep Dive for {country}")
        
        cols = st.columns(3)
        cols[0].metric("Official Language", language_analysis.get('language', 'N/A'))
        cols[1].metric("Learning Difficulty", f"{language_analysis.get('difficulty_score', 'N/A')}/5")
        cols[2].metric("Your Estimated Barrier", f"{int(language_analysis.get('estimated_barrier', 0))}/30")
        
        with st.expander("View Learning and Cultural Tips"):
            st.write(f"**Learning Tip:** {language_analysis.get('notes', 'No tips available.')}")
            
            # Placeholder for more detailed cultural tips
            st.write("**Cultural Integration Tip:** In many countries, trying to speak the local language, even poorly, is seen as a sign of respect and can help you build connections faster.")

def display_financial_analysis(country, profile):
    """Displays a detailed financial breakdown."""
    
    st.subheader(f"💰 Financial Readiness for {country}")

    # Load relevant data
    try:
        tuition_df = pd.read_csv('tuition_part_time.csv')
        cost_of_living_df = pd.read_csv('cost_of_living.csv')
    except FileNotFoundError as e:
        st.error(f"Financial data file missing: {e.filename}")
        return

    tuition_data = tuition_df[tuition_df['Country'] == country]
    cost_data = cost_of_living_df[cost_of_living_df['Country'] == country]

    if tuition_data.empty or cost_data.empty:
        st.warning(f"Complete financial data for {country} is not available.")
        return

    # Extract data
    avg_tuition = tuition_data.iloc[0].get('Average_Tuition_USD', 0)
    avg_cost_of_living = cost_data.iloc[0].get('Cost_of_Living_Index', 70) # Default to 70 if not found
    user_savings = profile.get('savings', 0)
    
    # Simple calculation for estimated first-year costs
    # This is a rough estimate and should be clearly marked as such
    estimated_first_year_cost = avg_tuition + (avg_cost_of_living * 300) # Simple multiplier for living costs

    savings_coverage = (user_savings / estimated_first_year_cost) * 100 if estimated_first_year_cost > 0 else 100

    cols = st.columns(2)
    cols[0].metric("Est. Avg. Tuition (USD)", f"${int(avg_tuition):,}")
    cols[1].metric("Est. 1st Year Cost (USD)", f"${int(estimated_first_year_cost):,}")

    st.progress(min(100, savings_coverage) / 100, text=f"Your savings cover ~{int(savings_coverage)}% of estimated first-year costs.")
    
    if savings_coverage < 50:
        st.error("High financial risk. Your savings may not be sufficient for the first year. Consider scholarships, loans, or part-time work.")
    elif savings_coverage < 100:
        st.warning("Moderate financial risk. You may need to supplement your savings. Look into part-time work opportunities.")
    else:
        st.success("Good financial standing. Your savings appear sufficient for initial costs.")

    # Part-time work analysis
    part_time_info = analyze_part_time_work(country)
    if part_time_info:
        with st.expander("View Part-Time Work Regulations for Students"):
            st.write(f"**Hours Allowed:** {part_time_info.get('hours_allowed', 'N/A')}")
            st.write(f"**Visa Type:** {part_time_info.get('visa_type', 'N/A')}")
            st.write(f"**Minimum Wage:** {part_time_info.get('minimum_wage_local', 'N/A')}")
            st.write(f"**Notes:** {part_time_info.get('notes', 'N/A')}")


def get_country_recommendations(country, profile):
    """Generate detailed text recommendations for a country."""
    
    risk_score = calculate_risk_score(profile, country)
    opportunity_score = calculate_opportunity_score(profile, country)
    lifestyle_score = calculate_lifestyle_score(profile, country)
    
    # Define thresholds
    low_risk, high_risk = 30, 70
    low_opp, high_opp = 40, 70
    
    recommendation = f"### Analysis for {country}:\n\n"
    
    # Risk Analysis
    if risk_score > high_risk:
        recommendation += f"**🔴 High Risk ({int(risk_score)}/100):** This path presents significant challenges. Your profile may not be a strong match for {country}'s requirements, particularly concerning financial readiness or language barriers. Detailed planning is critical.\n"
    elif risk_score > low_risk:
        recommendation += f"**🟡 Moderate Risk ({int(risk_score)}/100):** There are some potential hurdles to consider for {country}. While manageable, you should prepare for factors like cost of living and visa processes.\n"
    else:
        recommendation += f"**🟢 Low Risk ({int(risk_score)}/100):** Your profile aligns well with the practical requirements for {country}. The transition appears to be relatively smooth based on your savings and skills.\n"
        
    # Opportunity Analysis
    if opportunity_score > high_opp:
        recommendation += f"**🟢 High Opportunity ({int(opportunity_score)}/100):** {country} offers excellent prospects in your field ({profile.get('career_field', 'N/A')}). The job market and educational institutions are highly favorable for your profile.\n"
    elif opportunity_score > low_opp:
        recommendation += f"**🟡 Moderate Opportunity ({int(opportunity_score)}/100):** There are good opportunities in {country}, but they may be competitive. Networking and further specialization could be beneficial.\n"
    else:
        recommendation += f"**🔴 Low Opportunity ({int(opportunity_score)}/100):** The market for your field in {country} may be limited. Thoroughly research specific job openings or academic programs before committing.\n"
        
    # Lifestyle Analysis
    recommendation += f"**🔵 Lifestyle Score ({int(lifestyle_score)}/100):** The lifestyle in {country} offers a unique blend of cultural, social, and natural experiences. This score reflects how well it might fit your preferences for work-life balance and activities.\n"

    return recommendation


def calculate_risk_score(profile, country):
    """Calculate risk score based on profile and country data."""
    try:
        # Load necessary data
        cost_of_living_df = pd.read_csv('cost_of_living.csv')
        language_df = pd.read_csv('language_difficulty.csv')
        
        country_cost = cost_of_living_df[cost_of_living_df['Country'] == country]
        country_lang = language_df[language_df['Country'] == country]
        
        # --- Financial Risk ---
        cost_index = country_cost['Cost_of_Living_Index'].iloc[0] if not country_cost.empty else 70
        savings = profile.get('savings', 10000)
        # Risk is high if cost is high and savings are low. Normalize to a 0-50 scale.
        financial_risk = (cost_index / 130) * 50 - (savings / 50000) * 50 # 130 is a rough max cost index
        financial_risk = max(0, min(50, financial_risk))

        # --- Language Barrier Risk ---
        lang_difficulty = country_lang['Difficulty_Score'].iloc[0] if not country_lang.empty else 3
        lang_skills = profile.get('language_skills', 5)
        # Risk is high if difficulty is high and skills are low. Normalize to a 0-50 scale.
        language_risk = (lang_difficulty / 5) * 25 - (lang_skills / 10) * 25
        language_risk = max(0, min(25, language_risk))
        
        # --- Visa & Stability Risk (Placeholder) ---
        # In a real app, this would come from a stability index or visa difficulty dataset
        stability_risk = {"USA": 5, "Canada": 3, "UK": 6, "Germany": 4, "Australia": 4, "Japan": 2}.get(country, 8)
        
        # --- Total Risk Score ---
        # Invert risk tolerance: high tolerance means user is okay with higher risk, so it should lower the score.
        risk_tolerance_factor = (10 - profile.get('risk_tolerance', 5)) / 10
        
        total_risk = (financial_risk + language_risk + stability_risk) * risk_tolerance_factor
        return min(100, max(0, total_risk))
    
    except Exception:
        return 50 # Default score on error


def calculate_opportunity_score(profile, country):
    """Calculate opportunity score based on profile and country data."""
    try:
        # Load necessary data
        hdi_df = pd.read_csv('hdi_ranking.csv')
        gdp_growth_df = pd.read_csv('gdp_growth.csv')
        job_market_df = pd.read_csv('job_market_data.csv')
        uni_data = pd.read_csv('university_rankings.csv')
        
        country_hdi = hdi_df[hdi_df['Country'] == country]
        country_gdp = gdp_growth_df[gdp_growth_df['Country'] == country]
        country_job = job_market_df[job_market_df['Country'] == country]
        
        # --- Economic Opportunity (Max 30 points) ---
        # Based on HDI and GDP Growth
        hdi_score = country_hdi['HDI'].iloc[0] * 15 if not country_hdi.empty else 0.7 * 15
        gdp_growth = country_gdp['GDP_Growth'].iloc[0] if not country_gdp.empty else 1.0
        gdp_score = min(15, gdp_growth * 5) # Scale growth to a max of 15 points
        economic_score = hdi_score + gdp_score

        # --- Career Opportunity (Max 40 points) ---
        career_field = profile.get('career_field', 'Technology').replace(' ', '_')
        job_demand_score = 0
        if not country_job.empty and career_field in country_job.columns:
            # Score is based on demand level (e.g., 1-10)
            job_demand_score = country_job[career_field].iloc[0] * 4 # Scale to 40
        
        # --- Education Opportunity (Max 30 points) ---
        education_level = profile.get('education', 'Bachelors')
        # Higher score if the person has a lower degree (more room for growth)
        # or if the country has top universities.
        
        # Count top 100 universities in the country
        top_unis = uni_data[uni_data['Country'] == country].head(100).shape[0]
        uni_score = min(20, top_unis * 2) # Max 20 points from having many top universities
        
        education_advancement_score = 0
        if education_level == 'High School':
            education_advancement_score = 10
        elif education_level == 'Bachelors':
            education_advancement_score = 5

        education_score = uni_score + education_advancement_score
        
        total_opportunity = economic_score + job_demand_score + education_score
        return min(100, max(0, total_opportunity))
    
    except Exception:
        return 50 # Default score on error


def display_life_satisfaction_metrics(country):
    """Displays detailed life satisfaction metrics for a country."""
    
    try:
        df = pd.read_csv('life_satisfaction_metrics.csv')
        country_data = df[df['Country'] == country]
        
        if not country_data.empty:
            data = country_data.iloc[0]
            st.subheader(f"🧘 Life Satisfaction in {country}")
            
            cols = st.columns(4)
            cols[0].metric("Work-Life Balance", f"{data.get('Work_Life_Balance_Score', 'N/A')}/10")
            cols[1].metric("Social Life", f"{data.get('Social_Life_Score', 'N/A')}/10")
            cols[2].metric("Cultural Activities", f"{data.get('Cultural_Activities_Score', 'N/A')}/10")
            cols[3].metric("Nature Access", f"{data.get('Nature_Access_Score', 'N/A')}/10")
            
            with st.expander("About these scores"):
                st.write("""
                These scores are representative values based on public datasets and surveys.
                - **Work-Life Balance:** Considers average working hours, vacation time, and parental leave policies.
                - **Social Life:** Reflects community engagement, safety, and social support networks.
                - **Cultural Activities:** Based on the availability and accessibility of museums, theaters, concerts, and historical sites.
                - **Nature Access:** Measures green space in cities, national park coverage, and air/water quality.
                """)
        else:
            st.info(f"Life satisfaction data for {country} is not available.")
            
    except FileNotFoundError:
        st.error("`life_satisfaction_metrics.csv` not found.")
    except Exception as e:
        st.error(f"Error displaying life satisfaction data: {e}")
        
# ------------------------- Database Functions -------------------------

# Use Streamlit's caching for database connections
@st.cache_resource
def init_connection():
    """Initialize a connection to the SQLite database."""
    try:
        conn = sqlite3.connect('pathwise_history.db', check_same_thread=False)
        return conn
    except sqlite3.Error as e:
        st.error(f"Database connection failed: {e}")
        return None

def init_db(connection):
    """Initialize the database table if it doesn't exist."""
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                age INTEGER,
                education TEXT,
                savings INTEGER,
                language_skills INTEGER,
                risk_tolerance INTEGER,
                career_field TEXT,
                tuition_importance INTEGER,
                predicted_country TEXT,
                risk_score REAL,
                opportunity_score REAL,
                lifestyle_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
    except sqlite3.Error as e:
        st.error(f"Database initialization failed: {e}")


def save_prediction(profile, country, scores):
    """Save a prediction and user profile to the database."""
    try:
        c.execute("""
            INSERT INTO predictions (
                age, education, savings, language_skills, risk_tolerance, career_field, 
                tuition_importance, predicted_country, risk_score, opportunity_score, lifestyle_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile.get('age'), profile.get('education'), profile.get('savings'),
            profile.get('language_skills'), profile.get('risk_tolerance'), profile.get('career_field'),
            profile.get('tuition_importance'), country, scores['risk'], scores['opportunity'], scores['lifestyle']
        ))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Failed to save analysis to history: {e}")

def get_history():
    """Retrieve the prediction history from the database."""
    try:
        return pd.read_sql_query("SELECT predicted_country, risk_score, opportunity_score, lifestyle_score, timestamp FROM predictions ORDER BY timestamp DESC", conn)
    except sqlite3.Error:
        st.error("Could not retrieve history from the database.")
        return pd.DataFrame()
        
# ------------------------- Main App UI -------------------------

# --- HEADER ---
# Load and display logo
try:
    # A more robust way to handle image paths
    logo_image = Image.open('logo.png')
    st.image(logo_image, width=200)
except FileNotFoundError:
    st.title("✈️ PathWise")

st.markdown("## Your Global Life Decision Assistant")
st.markdown("---")

# --- DATABASE INITIALIZATION ---
conn = init_connection()
if conn:
    c = conn.cursor()
    init_db(conn)

# --- SIDEBAR ---
with st.sidebar:
    st.header("👤 Your Profile")
    
    age = st.slider("Age", 18, 65, 25)
    education = st.selectbox("Highest Education", ["High School", "Bachelors", "Masters", "PhD"])
    savings = st.slider("Savings (USD)", 0, 100000, 20000, step=1000)
    language_skills = st.slider("Primary Language Skills (1-10)", 1, 10, 7)
    risk_tolerance = st.slider("Risk Tolerance (1-10)", 1, 10, 5)
    career_field = st.selectbox("Career Field", ["Technology", "Healthcare", "Finance", "Education", "Arts", "Trades"])
    tuition_importance = st.slider("How important is low tuition? (1-10)", 1, 10, 8)
    
    # Compile profile
    profile = {
        'age': age,
        'education': education,
        'savings': savings,
        'language_skills': language_skills,
        'risk_tolerance': risk_tolerance,
        'career_field': career_field,
        'tuition_importance': tuition_importance
    }

    st.header("🌍 Select Countries")
    # Add a feature to select all popular countries
    popular_countries_list = ["United States", "Canada", "United Kingdom", "Germany", "Australia", "Japan", "France", "New Zealand"]
    
    # Use a session state to manage country selection
    if 'selected_countries' not in st.session_state:
        st.session_state.selected_countries = ["Canada", "Germany", "Japan"]

    # UI for country selection
    selected_countries = st.multiselect(
        "Choose countries to compare",
        options=popular_countries_list,
        default=st.session_state.selected_countries
    )
    st.session_state.selected_countries = selected_countries


# --- MAIN CONTENT ---

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis Dashboard", "Deep Dive Comparison", "📄 History", "Batch Analysis"])

with tab1:
    st.header("Analysis Dashboard")
    
    if not selected_countries:
        st.warning("Please select at least one country from the sidebar to begin analysis.")
    else:
        # --- SCORE CALCULATION ---
        results = []
        for country in selected_countries:
            risk = calculate_risk_score(profile, country)
            opportunity = calculate_opportunity_score(profile, country)
            lifestyle = calculate_lifestyle_score(profile, country)
            results.append({'Country': country, 'Risk Score': risk, 'Opportunity Score': opportunity, 'Lifestyle Score': lifestyle})
            
            # Save each analysis to the database
            if conn:
                save_prediction(profile, country, {'risk': risk, 'opportunity': opportunity, 'lifestyle': lifestyle})

        results_df = pd.DataFrame(results)

        # --- RECOMMENDATION ---
        # Simple recommendation: lowest risk, highest opportunity
        if not results_df.empty:
            results_df['Overall Score'] = results_df['Opportunity Score'] * 1.2 + results_df['Lifestyle Score'] * 0.8 - results_df['Risk Score']
            best_country_row = results_df.loc[results_df['Overall Score'].idxmax()]
            best_country_name = best_country_row['Country']
            
            st.success(f"**Top Recommendation: {best_country_name}**")
            st.markdown(get_country_recommendations(best_country_name, profile))
        
        # --- VISUALIZATIONS ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Scores Overview")
            fig_bar = px.bar(results_df, x='Country', y=['Risk Score', 'Opportunity Score', 'Lifestyle Score'], 
                             barmode='group', title="Country Scores Comparison",
                             color_discrete_map={
                                 'Risk Score': '#EF553B',
                                 'Opportunity Score': '#636EFA',
                                 'Lifestyle Score': '#00CC96'
                             })
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("Risk vs. Opportunity")
            fig_scatter = px.scatter(results_df, x='Risk Score', y='Opportunity Score', text='Country',
                                     size='Lifestyle Score', color='Country',
                                     title="Risk vs. Opportunity Matrix")
            fig_scatter.update_traces(textposition='top center')
            st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.header("Deep Dive Comparison")
    
    if not selected_countries:
        st.warning("Please select countries in the sidebar to compare.")
    else:
        # Create a select box to choose a country for the deep dive
        dive_country = st.selectbox("Select a country to dive into", options=selected_countries)
        
        if dive_country:
            # Display detailed components for the selected country
            display_financial_analysis(dive_country, profile)
            st.markdown("---")
            display_language_requirements(dive_country, profile)
            st.markdown("---")
            display_life_satisfaction_metrics(dive_country)

with tab3:
    st.header("📄 Your Analysis History")
    history_df = get_history()
    
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
        
        # Add a button to clear the history
        if st.button("Clear History"):
            if clear_history():
                st.success("History cleared!")
                st.experimental_rerun() # Rerun to update the view
            else:
                st.error("Failed to clear history.")
    else:
        st.info("No past analyses found. Your results will be saved here automatically.")

with tab4:
    st.header("Batch Analysis via CSV Upload")
    st.write("Upload a CSV file with multiple profiles to analyze them all at once.")
    st.write("Required columns: `age`, `education`, `savings`, `language_skills`, `risk_tolerance`, `career_field`, `tuition_importance`, `countries` (comma-separated).")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        batch_results_df = process_uploaded_csv(uploaded_file)
        if batch_results_df is not None:
            st.success("Batch analysis complete!")
            st.dataframe(batch_results_df, use_container_width=True)
            
            # Allow downloading the results
            csv = batch_results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                 label="Download results as CSV",
                 data=csv,
                 file_name='batch_analysis_results.csv',
                 mime='text/csv',
            )
            
# ------------------------- Advanced Features (Sidebar) -------------------------
with st.sidebar:
    st.markdown("---")
    st.header("Advanced Features")

    # Trend Analysis
    with st.expander("🌍 View Global Trends"):
        trends = get_career_trends()
        if trends:
            selected_field = st.selectbox("Select a field to see trends", options=list(trends.keys()))
            st.write(f"**Top roles in {selected_field}:**")
            for role in trends[selected_field]:
                st.markdown(f"- {role}")
        
        avg_scores = get_average_scores()
        if avg_scores:
            st.write("**Average scores across all analyses:**")
            st.write(f"Risk: {avg_scores[0]:.1f}, Opportunity: {avg_scores[1]:.1f}, Lifestyle: {avg_scores[2]:.1f}")
            
    # PDF Report
    if st.button("Generate PDF Report"):
        if 'results_df' in locals() and not results_df.empty:
            pdf_data = generate_pdf_report(profile, results_df[['Country', 'Risk Score', 'Opportunity Score', 'Lifestyle Score']], get_country_recommendations(best_country_name, profile))
            if pdf_data:
                st.download_button(
                    label="Download Report",
                    data=pdf_data,
                    file_name="PathWise_Report.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Please run an analysis first before generating a report.")

# ------------------------- Styling and Footer -------------------------
# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .stApp {
        background-color: #f9f9f9;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)
# Custom CSS
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #F0F2F6;
    }

    /* Sidebar styling */
    .st-emotion-cache-1y4p8pa {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Main content styling */
    .st-emotion-cache-16txtl3 {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin: 10px 0;
    }

    /* Tab styling */
    .st-emotion-cache-1sgoq9v.stTabs button {
        color: #555;
        font-weight: 500;
    }
    .st-emotion-cache-1sgoq9v.stTabs button[aria-selected="true"] {
        color: #636EFA;
        border-bottom-color: #636EFA;
    }
    
    /* Expander styling */
    .st-emotion-cache-p5msec {
        border-color: #E0E0E0;
        border-radius: 8px;
    }

</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>PathWise © 2024 - Your guide to a global future.</div>", unsafe_allow_html=True)


# Custom CSS for better mobile responsiveness
st.markdown("""
    <style>
    @media (max-width: 768px) {
        /* Reduce padding on mobile */
        .st-emotion-cache-16txtl3 {
            padding: 1rem;
        }
        
        /* Ensure text is readable */
        .st-emotion-cache-16txtl3, .st-emotion-cache-1y4p8pa {
            font-size: 14px;
        }

        /* Adjust button sizes */
        .stButton>button {
            width: 100%;
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
