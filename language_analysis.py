import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Read the CSV file
df = pd.read_csv('language_culture_comparison.csv')

def analyze_japanese_progression():
    """Analyze Japanese language progression path"""
    japan_data = df[df['Country'] == 'Japan'].copy()
    
    # Create progression visualization
    fig = go.Figure()
    
    # Add traces for different requirements
    fig.add_trace(go.Scatter(
        x=japan_data['Language_Level'],
        y=japan_data['Business_Requirement'],
        name='Business',
        line=dict(color='#3B82F6', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=japan_data['Language_Level'],
        y=japan_data['Tech_Requirement'],
        name='Tech',
        line=dict(color='#10B981', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=japan_data['Language_Level'],
        y=japan_data['Education_Requirement'],
        name='Education',
        line=dict(color='#F59E0B', width=2)
    ))
    
    fig.update_layout(
        title='Japanese Language Progression Path',
        xaxis_title='JLPT Level',
        yaxis_title='Requirement Level (%)',
        template='plotly_dark',
        hovermode='x unified'
    )
    
    return fig

def compare_work_culture():
    """Compare work culture across countries"""
    work_culture_data = df.drop_duplicates(subset=['Country'])[
        ['Country', 'Work_Culture_Score', 'Work_Life_Balance', 'International_Friendly']
    ]
    
    fig = go.Figure()
    
    # Add bars for each metric
    fig.add_trace(go.Bar(
        name='Work Culture',
        x=work_culture_data['Country'],
        y=work_culture_data['Work_Culture_Score'],
        marker_color='#3B82F6'
    ))
    
    fig.add_trace(go.Bar(
        name='Work-Life Balance',
        x=work_culture_data['Country'],
        y=work_culture_data['Work_Life_Balance'],
        marker_color='#10B981'
    ))
    
    fig.add_trace(go.Bar(
        name='International Friendly',
        x=work_culture_data['Country'],
        y=work_culture_data['International_Friendly'],
        marker_color='#F59E0B'
    ))
    
    fig.update_layout(
        title='Work Culture Comparison',
        barmode='group',
        template='plotly_dark',
        yaxis_title='Score (out of 10)'
    )
    
    return fig

def analyze_study_costs():
    """Analyze language study costs and duration"""
    cost_data = df.drop_duplicates(subset=['Country'])[
        ['Country', 'Language_School_Count', 'Min_Study_Period', 'Avg_Cost_Monthly_USD']
    ]
    
    fig = go.Figure()
    
    # Add scatter plot with size representing school count
    fig.add_trace(go.Scatter(
        x=cost_data['Min_Study_Period'],
        y=cost_data['Avg_Cost_Monthly_USD'],
        mode='markers+text',
        marker=dict(
            size=cost_data['Language_School_Count'] / 10,
            color=cost_data['Avg_Cost_Monthly_USD'],
            colorscale='Viridis',
            showscale=True
        ),
        text=cost_data['Country'],
        textposition='top center'
    ))
    
    fig.update_layout(
        title='Language Study Cost Analysis',
        xaxis_title='Minimum Study Period (months)',
        yaxis_title='Average Monthly Cost (USD)',
        template='plotly_dark'
    )
    
    return fig

def main():
    st.title('🌏 Global Language & Work Culture Analysis')
    
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1rem 0;'>
            <h3 style='color: #60a5fa;'>Japanese Language Learning Path</h3>
            <p style='color: white;'>
                Track the progression from N5 to N1 and understand requirements across different sectors
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display Japanese progression analysis
    st.plotly_chart(analyze_japanese_progression(), use_container_width=True)
    
    # Display work culture comparison
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1rem 0;'>
            <h3 style='color: #60a5fa;'>Work Culture Comparison</h3>
            <p style='color: white;'>
                Compare work culture, work-life balance, and international friendliness across countries
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(compare_work_culture(), use_container_width=True)
    
    # Display study cost analysis
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1rem 0;'>
            <h3 style='color: #60a5fa;'>Language Study Cost Analysis</h3>
            <p style='color: white;'>
                Compare study costs, duration, and language school availability across countries
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(analyze_study_costs(), use_container_width=True)
    
    # Additional information about Japanese language study
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1rem 0;'>
            <h3 style='color: #60a5fa;'>Japanese Language Study Guide</h3>
            
            <h4 style='color: #93c5fd;'>JLPT Levels and Timeline</h4>
            <ul style='color: white;'>
                <li><strong>N5 (Beginner)</strong>: 1-2 months intensive study</li>
                <li><strong>N4 (Basic)</strong>: 2-3 months additional study</li>
                <li><strong>N3 (Intermediate)</strong>: 3-4 months additional study</li>
                <li><strong>N2 (Business)</strong>: 4-6 months additional study</li>
                <li><strong>N1 (Advanced)</strong>: 6+ months additional study</li>
            </ul>
            
            <h4 style='color: #93c5fd;'>Recommended Study Methods</h4>
            <ul style='color: white;'>
                <li>Combine intensive courses with self-study</li>
                <li>Use language exchange apps (HelloTalk, Tandem)</li>
                <li>Practice with native speakers regularly</li>
                <li>Focus on business terminology in your field</li>
                <li>Utilize company-provided language training</li>
            </ul>
            
            <h4 style='color: #93c5fd;'>Useful Resources</h4>
            <ul style='color: white;'>
                <li>JLPT Official Website: Practice tests and study materials</li>
                <li>Japan Foundation: Free online courses and resources</li>
                <li>GaijinPot: Job listings and study guides</li>
                <li>Language school directories and reviews</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 