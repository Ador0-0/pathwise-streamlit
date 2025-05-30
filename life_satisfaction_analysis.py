import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Read the CSV file and filter out the Happiness_Factors line
df = pd.read_csv('life_satisfaction_metrics.csv', skipfooter=1, engine='python')

def analyze_happiness_rankings():
    """Create a comprehensive happiness ranking visualization"""
    fig = go.Figure()
    
    # Add happiness score bars
    fig.add_trace(go.Bar(
        x=df['Country'],
        y=df['Life_Satisfaction_Score'],
        name='Life Satisfaction',
        marker_color='#3B82F6'
    ))
    
    # Add stress level line (inverted for better visualization)
    fig.add_trace(go.Scatter(
        x=df['Country'],
        y=10 - df['Stress_Level'],  # Invert stress level
        name='Low Stress',
        yaxis='y2',
        line=dict(color='#F59E0B', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title='Global Happiness Rankings and Stress Levels',
        yaxis=dict(title='Life Satisfaction Score'),
        yaxis2=dict(
            title='Stress Level (Inverted)',
            overlaying='y',
            side='right'
        ),
        template='plotly_dark',
        barmode='group',
        height=600
    )
    
    return fig

def analyze_work_life_balance():
    """Analyze work-life balance metrics"""
    fig = go.Figure()
    
    # Add traces for different metrics
    fig.add_trace(go.Bar(
        name='Work-Life Balance',
        x=df['Country'],
        y=df['Work_Life_Balance_Score'],
        marker_color='#3B82F6'
    ))
    
    fig.add_trace(go.Bar(
        name='Leisure Time (hrs/week)',
        x=df['Country'],
        y=df['Leisure_Time_Hours_Weekly'],
        marker_color='#10B981'
    ))
    
    fig.add_trace(go.Bar(
        name='Paid Vacation Days',
        x=df['Country'],
        y=df['Paid_Vacation_Days'],
        marker_color='#F59E0B'
    ))
    
    fig.update_layout(
        title='Work-Life Balance Analysis',
        barmode='group',
        template='plotly_dark',
        height=500
    )
    
    return fig

def create_life_quality_radar():
    """Create a radar chart for life quality metrics"""
    # Select top 5 countries by Life Satisfaction Score
    top_countries = df.nlargest(5, 'Life_Satisfaction_Score')
    
    categories = ['Life_Satisfaction_Score', 'Work_Life_Balance_Score', 
                 'Mental_Health_Index', 'Social_Life_Score', 
                 'Nature_Access_Score', 'Cultural_Activities_Score']
    
    fig = go.Figure()
    
    for country in top_countries['Country']:
        country_data = top_countries[top_countries['Country'] == country]
        
        fig.add_trace(go.Scatterpolar(
            r=country_data[categories].values[0],
            theta=categories,
            fill='toself',
            name=country
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title='Life Quality Comparison (Top 5 Countries)',
        template='plotly_dark',
        height=500
    )
    
    return fig

def main():
    st.title('🌍 Global Life Satisfaction Analysis')
    
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1rem 0;'>
            <h3 style='color: #60a5fa;'>Global Happiness Rankings</h3>
            <p style='color: white;'>
                Explore life satisfaction scores and stress levels across different countries
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display happiness rankings
    st.plotly_chart(analyze_happiness_rankings(), use_container_width=True)
    
    # Display work-life balance analysis
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1rem 0;'>
            <h3 style='color: #60a5fa;'>Work-Life Balance Analysis</h3>
            <p style='color: white;'>
                Compare work-life balance metrics including leisure time and vacation days
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(analyze_work_life_balance(), use_container_width=True)
    
    # Display life quality radar chart
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1rem 0;'>
            <h3 style='color: #60a5fa;'>Life Quality Comparison</h3>
            <p style='color: white;'>
                Radar chart comparing various life quality metrics for top 5 happiest countries
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(create_life_quality_radar(), use_container_width=True)
    
    # Display detailed country information
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5);
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1rem 0;'>
            <h3 style='color: #60a5fa;'>Country Details</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Country selector
    selected_country = st.selectbox(
        "Select a country to view detailed information",
        df['Country'].tolist()
    )
    
    if selected_country:
        country_data = df[df['Country'] == selected_country].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("World Happiness Rank", f"#{int(country_data['World_Happiness_Rank'])}")
            st.metric("Life Satisfaction Score", f"{country_data['Life_Satisfaction_Score']:.2f}")
            st.metric("Mental Health Index", f"{country_data['Mental_Health_Index']:.1f}")
        
        with col2:
            st.metric("Work-Life Balance", f"{country_data['Work_Life_Balance_Score']:.1f}/10")
            st.metric("Leisure Time", f"{int(country_data['Leisure_Time_Hours_Weekly'])} hrs/week")
            st.metric("Vacation Days", f"{int(country_data['Paid_Vacation_Days'])} days")
        
        with col3:
            st.metric("Social Life Score", f"{country_data['Social_Life_Score']:.1f}/10")
            st.metric("Nature Access", f"{country_data['Nature_Access_Score']:.1f}/10")
            st.metric("Cultural Activities", f"{country_data['Cultural_Activities_Score']:.1f}/10")

if __name__ == "__main__":
    main() 