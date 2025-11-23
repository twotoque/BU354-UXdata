import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import numpy as np

YEAR_ORDER = ['Year 1', 'Year 2', 'Year 3', 'Year 4']
COLOURS = ['#5D3A9B', '#f6b118', '#c70c28', '#53b447', '#9370DB', '#FF8C00']
RANKING_COLS = [
    'Rank which UI you liked the best [Dashboard A]',
    'Rank which UI you liked the best [Dashboard B]',
    'Rank which UI you liked the best [Dashboard C]',
    'Rank which UI you liked the best [Dashboard D]',
    'Rank which UI you liked the best [Dashboard E]',
    'Rank which UI you liked the best [Dashboard F]'
]

def calculate_average_rankings(df):
    """
    Calculate average rankings for each dashboard from survey data.
    
    Args:
        df: DataFrame containing survey responses with ranking columns
        
    Returns:
        dict: Dictionary with dashboard names as keys and average rankings as values
    """
    
    avg_rankings = {}
    
    for col in RANKING_COLS:
        dashboard = col.split('[Dashboard ')[1].split(']')[0]
        
        numeric_rankings = df[col].astype(str).str.extract(r'(\d+)')[0].astype(float)
        
        avg_rankings[f'Dashboard {dashboard}'] = numeric_rankings.mean()
    
    return avg_rankings


def get_ranking_distributions(df):
    """
    Get the distribution of rankings for each dashboard.
    
    Args:
        df: DataFrame containing survey responses with ranking columns
        
    Returns:
        dict: Dictionary with dashboard names as keys and lists of rankings as values
    """
    
    distributions = {}
    
    for col in RANKING_COLS:
        dashboard = col.split('[Dashboard ')[1].split(']')[0]
        
        numeric_rankings = df[col].astype(str).str.extract(r'(\d+)')[0].astype(float)
        
        distributions[f'Dashboard {dashboard}'] = numeric_rankings.dropna().tolist()
    
    return distributions

def render_year_distribution(df, output_filename='year_distribution.pdf'):
    """
    Render a bar chart showing the distribution of academic years in the survey.
    
    Args:
        df: DataFrame containing survey responses
        output_filename: Name of the output PDF file
    """
    year_col = 'What academic year are you in?'
    
    year_counts = df[year_col].value_counts()
    
    year_counts = year_counts.reindex([y for y in YEAR_ORDER if y in year_counts.index])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=year_counts.index,
        y=year_counts.values,
        marker=dict(
            color=COLOURS[:len(year_counts)],
            line=dict(color='white', width=1)
        ),
        text=year_counts.values,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Survey Participation by Academic Year<br><sub>Number of respondents per year</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Academic Year',
        yaxis_title='Number of Respondents',
        font=dict(size=12),
        showlegend=False,
        height=600,
        width=1000,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_xaxes(showgrid=False)
    
    fig.show()
    
    fig.write_image(output_filename, width=1000, height=600)
    
    return fig

def compare_two_dashboards(df, dashboard1='A', dashboard2='E', output_filename='ae-t-test.pdf'):
    """
    Perform paired t-test between two dashboards.
    
    Args:
        df: DataFrame containing survey responses
        dashboard1: Letter of first dashboard (e.g., 'A')
        dashboard2: Letter of second dashboard (e.g., 'E')
    """
    col1 = f'Rank which UI you liked the best [Dashboard {dashboard1}]'
    col2 = f'Rank which UI you liked the best [Dashboard {dashboard2}]'
    
    rankings1 = df[col1].astype(str).str.extract(r'(\d+)')[0].astype(float).dropna()
    rankings2 = df[col2].astype(str).str.extract(r'(\d+)')[0].astype(float).dropna()
    
    t_stat, p_value = stats.ttest_rel(rankings1, rankings2)
    
    mean1 = rankings1.mean()
    mean2 = rankings2.mean()
    mean_diff = mean1 - mean2
    
    std_diff = (rankings1 - rankings2).std()
    cohens_d = mean_diff / std_diff if std_diff != 0 else 0
    
    print(f"\nPaired T-Test: Dashboard {dashboard1} vs Dashboard {dashboard2}")
    print("=" * 50)
    print(f"Dashboard {dashboard1} - Mean Rank: {mean1:.2f}")
    print(f"Dashboard {dashboard2} - Mean Rank: {mean2:.2f}")
    print(f"Mean Difference: {mean_diff:.2f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Cohen's d (effect size): {cohens_d:.3f}")
    
    if p_value < 0.05:
        better = dashboard1 if mean1 < mean2 else dashboard2
        print(f"\n✓ Result: Dashboard {better} is significantly preferred (p < 0.05)")
    else:
        print(f"\n✗ Result: No significant difference (p ≥ 0.05)")
    
    fig = go.Figure()
    
    # Side-by-side box plots
    fig.add_trace(go.Box(
        y=rankings1,
        name=f'Dashboard {dashboard1}',
        marker=dict(color='#5D3A9B'),
        boxmean='sd'
    ))
    
    fig.add_trace(go.Box(
        y=rankings2,
        name=f'Dashboard {dashboard2}',
        marker=dict(color='#53b447'),
        boxmean='sd'
    ))
    
    if p_value < 0.05:
        significance_text = f"p = {p_value:.4f} *<br>Cohen's d = {cohens_d:.2f}"
    else:
        significance_text = f"p = {p_value:.4f} (n.s.)<br>Cohen's d = {cohens_d:.2f}"
    
    fig.update_layout(
        title=dict(
            text=f'Dashboard {dashboard1} vs Dashboard {dashboard2}: Paired T-Test<br><sub>{significance_text}</sub>',
            x=0.5,
            xanchor='center'
        ),
        yaxis_title='Ranking (1 = Most Liked, 6 = Least Liked)',
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 6.5]
        ),
        font=dict(size=12),
        showlegend=True,
        height=600,
        width=800,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    fig.show()
    fig.write_image(output_filename, width=800, height=600)

    return t_stat, p_value, cohens_d

def render_rankings_graph(avg_rankings, output_filename='dashboard_rankings.pdf'):
    """
    Render a Plotly bar graph of dashboard rankings and export to PDF.
    
    Args:
        avg_rankings: Dictionary with dashboard names and average rankings
        output_filename: Name of the output PDF file
    """
    dashboards = list(avg_rankings.keys())
    averages = list(avg_rankings.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dashboards,
        y=averages,
        marker=dict(
            color=averages,
            colorscale='RdYlGn_r',  # Red for high (worst), Green for low (best)
            showscale=True,
            colorbar=dict(title="Avg Rank")
        ),
        text=[f'{avg:.2f}' for avg in averages],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Average Rank: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Average UI Ranking by Dashboard<br><sub>Lower is Better (1 = Most Liked, 6 = Least Liked)</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Dashboard',
        yaxis_title='Average Ranking',
        yaxis=dict(
            title='Average Ranking (Lower = Better)'
        ),
        font=dict(size=12),
        showlegend=False,
        height=600,
        width=1000,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    fig.show()
    
    fig.write_image(output_filename, width=1000, height=600)
    
    return fig


def render_major_distribution(df, output_filename='major_distribution.pdf'):
    """
    Render a bar chart showing the distribution of majors in the survey.
    
    Args:
        df: DataFrame containing survey responses
        output_filename: Name of the output PDF file
    """
    major_col = 'What best describes your major?'
    
    major_counts = df[major_col].value_counts().sort_values(ascending=False)
    
    major_labels = [major.split(',')[0] if ',' in major else major for major in major_counts.index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=major_labels,
        y=major_counts.values,
        marker=dict(
            color=COLOURS * (len(major_counts) // len(COLOURS) + 1),
            line=dict(color='white', width=1)
        ),
        text=major_counts.values,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Survey Participation by Major<br><sub>Number of respondents per major</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Major',
        yaxis_title='Number of Respondents',
        font=dict(size=12),
        showlegend=False,
        height=600,
        width=1000,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_xaxes(showgrid=False)
    
    fig.show()
    
    fig.write_image(output_filename, width=1000, height=600)
    
    return fig

def get_rankings_by_major(df):
    """
    Get rankings grouped by major for analysis.
    
    Args:
        df: DataFrame containing survey responses
        
    Returns:
        dict: Nested dictionary {dashboard: {major: [rankings]}}
    """
    
    major_col = 'What best describes your major?'
    
    rankings_by_major = {}
    
    for col in RANKING_COLS:
        dashboard = col.split('[Dashboard ')[1].split(']')[0]
        dashboard_name = f'Dashboard {dashboard}'
        rankings_by_major[dashboard_name] = {}
        
        # Group by major
        for idx, row in df.iterrows():
            major = row[major_col]
            ranking_str = str(row[col])
            
            # Extract numeric ranking
            import re
            match = re.search(r'(\d+)', ranking_str)
            if match and pd.notna(major):
                ranking = float(match.group(1))
                
                if major not in rankings_by_major[dashboard_name]:
                    rankings_by_major[dashboard_name][major] = []
                
                rankings_by_major[dashboard_name][major].append(ranking)
    
    return rankings_by_major


def render_major_impact_scatter(df, output_filename='dashboard_by_major_scatter.pdf'):
    """
    Render scatter plots showing how major affects dashboard rankings.
    
    Args:
        df: DataFrame containing survey responses
        output_filename: Name of the output PDF file
    """
    rankings_by_major = get_rankings_by_major(df)

    fig = go.Figure()

    # Collect majors and dashboards
    all_majors = set()
    for dashboard_data in rankings_by_major.values():
        all_majors.update(dashboard_data.keys())
    majors_list = sorted(list(all_majors))
    dashboards = list(rankings_by_major.keys())

    # Compute averages and add bar traces
    for i, major in enumerate(majors_list):
        colour = COLOURS[i % len(COLOURS)]
        major_short = major.split(',')[0] if ',' in major else major

        y_values = [
            (sum(ranks) / len(ranks)) if ranks else None
            for ranks in [rankings_by_major[d].get(major, []) for d in dashboards]
        ]

        fig.add_trace(go.Bar(
            x=dashboards,
            y=y_values,
            name=major_short,
            marker_color=colour,
            opacity=0.8
        ))


    # Layout settings
    fig.update_layout(
        title=dict(
            text='Dashboard Rankings by Major<br><sub>Lined Grouped Bar Chart (Lower = Better)</sub>',
            x=0.5,
            xanchor='center'
        ),
        barmode='group',
        xaxis_title='Dashboard',
        yaxis_title='Average Ranking (1 = Most Liked, 6 = Least Liked)',
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 6.5]
        ),
        font=dict(size=11),
        height=700,
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            title='Major',
            orientation='h',
            yanchor='top',
            y=-0.2,      
            xanchor='center',
            x=0.5
        )
    )

    # Grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    fig.show()
    fig.write_image(output_filename, width=1200, height=700)
    return fig


def render_major_impact_grouped(df, output_filename='dashboard_by_major_grouped.pdf'):
    """
    Render grouped box plots showing dashboard rankings distribution by major.
    
    Args:
        df: DataFrame containing survey responses
        output_filename: Name of the output PDF file
    """
    rankings_by_major = get_rankings_by_major(df)
    
    fig = go.Figure()
    
    # Get unique majors
    all_majors = set()
    for dashboard_data in rankings_by_major.values():
        all_majors.update(dashboard_data.keys())
    
    majors_list = sorted(list(all_majors))

    for i, major in enumerate(majors_list):
        colour = COLOURS[i % len(COLOURS)]
        major_short = major.split(',')[0] if ',' in major else major
        
        for dashboard, major_data in rankings_by_major.items():
            if major in major_data:
                fig.add_trace(go.Box(
                    y=major_data[major],
                    name=major_short,
                    x=[dashboard] * len(major_data[major]),
                    marker=dict(color=colour),
                    legendgroup=major_short,
                    showlegend=(dashboard == 'Dashboard A'),  # Show legend only once per major
                    boxmean=True
                ))
    
    fig.update_layout(
        title=dict(
            text='Dashboard Rankings Distribution by Major<br><sub>Grouped box plots (Lower = Better)</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Dashboard',
        yaxis_title='Ranking (1 = Most Liked, 6 = Least Liked)',
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 6.5]
        ),
        font=dict(size=11),
        height=700,
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        boxmode='group',
        legend=dict(
            title='Major',
            orientation='h',
            yanchor='top',
            y=-0.2,      
            xanchor='center',
            x=0.5
        )
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_xaxes(showgrid=False)
    
    fig.show()
    fig.write_image(output_filename, width=1200, height=700)
    
    return fig

def render_ranking_distribution(distributions, output_filename='dashboard_distribution.pdf'):
    """
    Render a box plot showing the distribution of rankings for each dashboard.
    
    Args:
        distributions: Dictionary with dashboard names and lists of rankings
        output_filename: Name of the output PDF file
    """
    fig = go.Figure()
    
    # Add a box plot for each dashboard
    for dashboard, rankings in distributions.items():
        fig.add_trace(go.Box(
            y=rankings,
            name=dashboard,
            boxmean='sd',  
            marker=dict(
                color='rgba(93, 58, 155, 0.6)',
                line=dict(color='rgb(93, 58, 155)', width=2)
            ),
            line=dict(color='rgb(93, 58, 155)')
        ))
    
    fig.update_layout(
        title=dict(
            text='Distribution of Rankings per Dashboard<br><sub>Box plots showing spread and outliers (Lower = Better)</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Dashboard',
        yaxis_title='Ranking (1 = Most Liked, 6 = Least Liked)',
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 6.5]
        ),
        font=dict(size=12),
        showlegend=False,
        height=600,
        width=1000,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_xaxes(showgrid=False)
    
    fig.show()
    
    fig.write_image(output_filename, width=1000, height=600)
    
    return fig

def chi_square_preference_test(df, column_name, expected_proportions=None):
    """
    Perform chi-square goodness of fit test for preference questions.
    
    Args:
        df: DataFrame containing survey responses
        column_name: Name of the column with categorical responses
        expected_proportions: Expected distribution (None = equal proportions)
    
    Returns:
        chi2_stat, p_value, observed_counts, expected_counts
    """
    observed_counts = df[column_name].value_counts().sort_index()
    categories = observed_counts.index.tolist()
    observed = observed_counts.values
    
    n = observed.sum()
    if expected_proportions is None:
        # assume equal proportions (e.g., 50/50 for two options)
        expected = np.array([n / len(categories)] * len(categories))
    else:
        expected = np.array([n * p for p in expected_proportions])
    
    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    
    print(f"\nChi-Square Goodness of Fit Test")
    print("=" * 50)
    print(f"Question: {column_name}")
    print(f"\nObserved vs Expected Counts:")
    print("-" * 50)
    
    for i, category in enumerate(categories):
        percentage = (observed[i] / n) * 100
        print(f"{category}:")
        print(f"  Observed: {observed[i]} ({percentage:.1f}%)")
        print(f"  Expected: {expected[i]:.1f}")
    
    print(f"\nChi-square statistic: {chi2_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Degrees of freedom: {len(categories) - 1}")
    
    if p_value < 0.05:
        print(f"\n✓ Result: Preferences are significantly different from expected (p < 0.05)")
    else:
        print(f"\n✗ Result: Preferences are not significantly different from expected (p ≥ 0.05)")
    
    cramers_v = np.sqrt(chi2_stat / n)
    print(f"Cramér's V (effect size): {cramers_v:.3f}")
    
    return chi2_stat, p_value, observed, expected


def chi_square_independence_test(df, column1, column2):
    """
    Perform chi-square test of independence between two categorical variables.
    
    Args:
        df: DataFrame containing survey responses
        column1: First categorical variable (e.g., 'navigation preference')
        column2: Second categorical variable (e.g., 'major')
    
    Returns:
        chi2_stat, p_value, contingency_table
    """
    contingency_table = pd.crosstab(df[column1], df[column2])
    
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\nChi-Square Test of Independence")
    print("=" * 50)
    print(f"Testing: {column1} vs {column2}")
    print(f"\nContingency Table:")
    print(contingency_table)
    
    print(f"\nChi-square statistic: {chi2_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    
    if p_value < 0.05:
        print(f"\n✓ Result: Variables are significantly related (p < 0.05)")
    else:
        print(f"\n✗ Result: Variables are independent (p ≥ 0.05)")
    
    # cramers v 
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape[0], contingency_table.shape[1])
    cramers_v = np.sqrt(chi2_stat / (n * (min_dim - 1)))
    print(f"Cramér's V (effect size): {cramers_v:.3f}")
    
    return chi2_stat, p_value, contingency_table

def visualize_chi_square(df, column_name, output_filename='chi_square_test.pdf'):
    """
    Visualize categorical preference data with chi-square test results.
    
    Args:
        df: DataFrame containing survey responses
        column_name: Name of the column with categorical responses
        output_filename: Name of the output PDF file
    """
    counts = df[column_name].value_counts().sort_index()
    categories = counts.index.tolist()
    observed = counts.values
    
    n = observed.sum()
    expected = np.array([n / len(categories)] * len(categories))
    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=observed,
        name='Observed',
        marker=dict(color='#5D3A9B'),
        text=observed,
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=categories,
        y=expected,
        name='Expected (under H₀)',
        marker=dict(color='#f6b118', opacity=0.6),
        text=[f'{e:.1f}' for e in expected],
        textposition='outside'
    ))
    
    if p_value < 0.05:
        sig_text = f"χ² = {chi2_stat:.2f}, p = {p_value:.4f} *<br>H₀: Equal preference | H₁: Unequal preference | Result: Reject H₀"
    else:
        sig_text = f"χ² = {chi2_stat:.2f}, p = {p_value:.4f} (n.s.)<br>H₀: Equal preference | H₁: Unequal preference | Result: Fail to reject H₀"
    
    fig.update_layout(
        title=dict(
            text=f'{column_name}<br><sub>{sig_text}</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Response',
        yaxis_title='Count',
        font=dict(size=12),
        showlegend=True,
        height=600,
        width=800,
        plot_bgcolor='white',
        paper_bgcolor='white',
        barmode='group'
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    fig.show()
    fig.write_image(output_filename, width=800, height=600)
    
    print(f"\nVisualization saved to: {output_filename}")
    
    return fig
    
def main():
    """Main function to orchestrate the ranking analysis."""
    googleForm = pd.read_csv('/Users/dereksong/Documents/Newsletter Spring Front/Spring Newsletter Jamaal Myers/img/oGlogos/winternship-website/bu354-platform/ux-research/UXsurvey.csv')
    totalResearch = pd.read_csv('/Users/dereksong/Documents/Newsletter Spring Front/Spring Newsletter Jamaal Myers/img/oGlogos/winternship-website/bu354-platform/ux-research/UXsurveyStickyNote.csv')

    avgRanking = calculate_average_rankings(googleForm)
    avgRankingtotalResearch = calculate_average_rankings(totalResearch)

    distributions = get_ranking_distributions(googleForm)
    distributionstotalResearch = get_ranking_distributions(totalResearch)

    # render/export
    print("Generating average rankings bar chart...")
    render_rankings_graph(avgRanking, './ux-research/pdfs/dashboard_rankings.pdf')
    print("Generating ranking distribution box plot...")
    render_ranking_distribution(distributions, './ux-research/pdfs/dashboard_distribution.pdf')

    print("Generating average rankings bar chart (total)...")
    render_rankings_graph(avgRankingtotalResearch, './ux-research/pdfs/dashboard_rankings_total.pdf')
    print("Generating ranking distribution box plot (total)...")
    render_ranking_distribution(distributionstotalResearch, './ux-research/pdfs/dashboard_distribution_total.pdf')

    print("Generating major impact scatter plot...")
    render_major_impact_scatter(googleForm, './ux-research/pdfs/dashboard_by_major_scatter.pdf')
    print("Generating major impact grouped box plot...")
    render_major_impact_grouped(googleForm, './ux-research/pdfs/dashboard_by_major_grouped.pdf')
    print("Generating major distribution bar chart...")
    render_major_distribution(googleForm, './ux-research/pdfs/major_distribution.pdf')
    print("Generating year distribution bar chart...")
    render_year_distribution(googleForm, './ux-research/pdfs/year_distribution.pdf')
    print("\nStatistical Comparison of Front-runners:")
    
    compare_two_dashboards(googleForm, 'A', 'E', './ux-research/pdfs/comparison_A_vs_E.pdf')
    compare_two_dashboards(googleForm, 'A', 'D', './ux-research/pdfs/comparison_A_vs_D.pdf')

    chi_square_preference_test(googleForm, 'Which bar style do you prefer?')
    visualize_chi_square(googleForm, 'Which bar style do you prefer?', './ux-research/pdfs/navigation_preference.pdf')

    chi_square_independence_test(googleForm,'Which bar style do you prefer?', 'What best describes your major?')

    # summaries
    print("Avg Rankings Summary:")
    print("=" * 40)
    for dashboard, avg in sorted(avgRanking.items(), key=lambda x: x[1]):
        print(f"{dashboard}: {avg:.2f}")
    
    print("\n" + "=" * 40)
    print("Distribution Statistics:")
    print("=" * 40)
    for dashboard, rankings in distributions.items():
        import statistics
        std_dev = statistics.stdev(rankings) if len(rankings) > 1 else 0
        median = statistics.median(rankings)
        print(f"{dashboard}:")
        print(f"  Median: {median}")
        print(f"  Std Dev: {std_dev:.2f}")
        print(f"  Sample Size: {len(rankings)}")
    print("Rankings by Major:")
    print("=" * 40)
    rankings_by_major = get_rankings_by_major(googleForm)
    for dashboard in sorted(rankings_by_major.keys()):
        print(f"\n{dashboard}:")
        for major, rankings in rankings_by_major[dashboard].items():
            import statistics
            avg = statistics.mean(rankings)
            major_short = major.split(',')[0] if ',' in major else major
            print(f"  {major_short}: {avg:.2f} (n={len(rankings)})")
    

    print(f"\nPDFs saved:")
    print("  - dashboard_rankings.pdf")
    print("  - dashboard_distribution.pdf")
    print("  - dashboard_by_major_scatter.pdf")
    print("  - dashboard_by_major_grouped.pdf")
    print("  - major_distribution.pdf")
    print("  - year_distribution.pdf")
    print("  - dashboard_rankings_total.pdf")
    print("  - dashboard_distribution_total.pdf")
    print("  - comparison_A_vs_D.pdf")
    print("  - comparison_A_vs_E.pdf")
    print("  - navigation_preference.pdf")



if __name__ == "__main__":
    main()