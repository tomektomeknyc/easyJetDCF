import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_elements import elements, mui, html, dashboard
from streamlit_card import card

import altair as alt
import math
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.chart_container import chart_container
import random

class AdvancedVisualizations:
    """
    A class providing cutting-edge financial visualizations
    """

    def __init__(self, dcf_analyzer):
        """
        Initialize with a DCF Analyzer instance
        """
        self.dcf = dcf_analyzer
        self.variables = dcf_analyzer.variables
        self.color_palette = {
            'primary': '#2196F3',
            'secondary': '#FF9800',
            'tertiary': '#4CAF50',
            'quaternary': '#9C27B0',
            'negative': '#F44336',
            'positive': '#4CAF50',
            'neutral': '#9E9E9E',
            'background': '#F5F5F5',
            'grid': 'rgba(0,0,0,0.05)'
        }

        # Generate a smooth gradient palette
        self.gradient_palette = self._generate_gradient_palette()

    def _generate_gradient_palette(self, num_colors=20):
        """Generate a smooth gradient color palette"""
        colors = []
        for i in range(num_colors):
            # Create a gradient from blue to orange
            r = int(33 + (242 - 33) * i / (num_colors-1))
            g = int(150 + (153 - 150) * i / (num_colors-1))
            b = int(243 + (0 - 243) * i / (num_colors-1))
            colors.append(f'rgb({r},{g},{b})')
        return colors

    def format_currency(self, value):
        """Format a numeric value as currency"""
        if isinstance(value, str):
            return value
        if math.isnan(value):
            return "N/A"

        # For values in millions, format appropriately
        if abs(value) >= 1e6:
            return f"£{value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"£{value/1e3:.2f}K"
        else:
            return f"£{value:.2f}"

    def format_percentage(self, value):
        """Format a numeric value as percentage"""
        if isinstance(value, str):
            return value
        if math.isnan(value):
            return "N/A"
        return f"{value:.2f}%"

    def display_header_dashboard(self):
        """Display an executive dashboard header with key metrics"""
        st.markdown("""
        <style>
        .metric-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #1E88E5;
        }
        .metric-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: space-between;
        }
        .metric-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 15px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        </style>

        <div class="metric-header">EasyJet Financial Summary</div>
        """, unsafe_allow_html=True)

        # Create a modern metric dashboard with cards
        col1, col2, col3, col4 = st.columns(4)

        # Current metrics
        with col1:
            current_price = self.variables['current_share_price']
            with stylable_container(
                key="current_price_container",
                css_styles="""
                {
                    background-color: #E3F2FD;
                    border-radius: 10px;
                    padding: 15px;
                }
                """,
            ):
                st.metric(
                    label="Current Share Price",
                    value=f"£{current_price:.2f}",
                    delta=None,
                )

        with col2:
            multiples_price = self.variables['share_price_multiples']
            pct_diff = ((multiples_price / current_price) - 1) * 100 if current_price > 0 else 0
            with stylable_container(
                key="multiples_price_container",
                css_styles="""
                {
                    background-color: #FFF8E1;
                    border-radius: 10px;
                    padding: 15px;
                }
                """,
            ):
                st.metric(
                    label="DCF Multiples Price",
                    value=f"£{multiples_price:.2f}",
                    delta=f"{pct_diff:.1f}%",
                )

        with col3:
            perpetuity_price = self.variables['share_price_perpetuity']
            pct_diff = ((perpetuity_price / current_price) - 1) * 100 if current_price > 0 else 0
            with stylable_container(
                key="perpetuity_price_container",
                css_styles="""
                {
                    background-color: #E8F5E9;
                    border-radius: 10px;
                    padding: 15px;
                }
                """,
            ):
                st.metric(
                    label="DCF Perpetuity Price",
                    value=f"£{perpetuity_price:.2f}",
                    delta=f"{pct_diff:.1f}%",
                )

        with col4:
            wacc = self.variables['wacc'] * 100
            growth = self.variables['terminal_growth'] * 100
            with stylable_container(
                key="wacc_growth_container",
                css_styles="""
                {
                    background-color: #F3E5F5;
                    border-radius: 10px;
                    padding: 15px;
                }
                """,
            ):
                st.metric(
                    label="WACC / Growth",
                    value=f"{wacc:.2f}% / {growth:.2f}%",
                    delta=None,
                )

        # Apply styles to metric cards
        style_metric_cards(
            background_color="#FFFFFF",
            border_left_color="#1E88E5",
            border_color="#EEEEEE",
            box_shadow=True,
            border_size_px=1,
            border_left_width=5,
        )

    def display_enterprise_value_3d(self):
        """Display an advanced 3D visualization of enterprise value"""
        st.subheader("Enterprise Value Analysis - 3D View")

        # Get values
        ev_multiples = self.variables['ev_multiples']
        ev_perpetuity = self.variables['ev_perpetuity']

        # Create a more sophisticated 3D visualization
        with chart_container(use_container_width=True):
            # Create a figure with subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                subplot_titles=("Multiples Method EV", "Perpetuity Growth Method EV"),
                horizontal_spacing=0.05
            )

            # Generate grid data for 3D surface
            wacc_base = self.variables['wacc']
            growth_base = self.variables['terminal_growth']

            wacc_range = np.linspace(max(0.01, wacc_base * 0.7), wacc_base * 1.3, 20)
            growth_range = np.linspace(max(0.005, growth_base * 0.5), growth_base * 1.5, 20)

            wacc_grid, growth_grid = np.meshgrid(wacc_range, growth_range)

            # Calculate enterprise value surfaces
            ev_multiples_grid = np.zeros_like(wacc_grid)
            ev_perpetuity_grid = np.zeros_like(wacc_grid)

            for i in range(wacc_grid.shape[0]):
                for j in range(wacc_grid.shape[1]):
                    w = wacc_grid[i, j]
                    g = growth_grid[i, j]

                    # Simplified calculation for demonstration
                    ev_multiples_grid[i, j] = ev_multiples * (wacc_base / w) ** 0.7

                    # More sensitive to both WACC and growth
                    perpetuity_factor = (1 + g) / (w - g)
                    base_perpetuity = (1 + growth_base) / (wacc_base - growth_base)
                    ev_perpetuity_grid[i, j] = ev_perpetuity * (perpetuity_factor / base_perpetuity)

            # First surface - Multiples Method
            fig.add_trace(
                go.Surface(
                    z=ev_multiples_grid,
                    x=wacc_grid * 100,  # Convert to percentage
                    y=growth_grid * 100,  # Convert to percentage
                    colorscale='Blues',
                    showscale=False,
                    lighting=dict(
                        ambient=0.8,
                        diffuse=0.8,
                        fresnel=0.2,
                        specular=1,
                        roughness=0.5
                    ),
                    contours={
                        "x": {"show": True, "color":"#1E88E5", "width": 2},
                        "y": {"show": True, "color":"#1E88E5", "width": 2},
                        "z": {"show": True, "color":"#1E88E5", "width": 2}
                    },
                    hovertemplate=(
                        "<b>WACC</b>: %{x:.2f}%<br>" +
                        "<b>Growth</b>: %{y:.2f}%<br>" +
                        "<b>EV (Multiples)</b>: £%{z:.2f}M<br>" +
                        "<extra></extra>"
                    )
                ),
                row=1, col=1
            )

            # Add a marker for current values in first surface
            fig.add_trace(
                go.Scatter3d(
                    x=[wacc_base * 100],
                    y=[growth_base * 100],
                    z=[ev_multiples],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='circle'
                    ),
                    name='Current Multiples EV'
                ),
                row=1, col=1
            )

            # Second surface - Perpetuity Method
            fig.add_trace(
                go.Surface(
                    z=ev_perpetuity_grid,
                    x=wacc_grid * 100,  # Convert to percentage
                    y=growth_grid * 100,  # Convert to percentage
                    colorscale='Oranges',
                    showscale=False,
                    lighting=dict(
                        ambient=0.8,
                        diffuse=0.8,
                        fresnel=0.2,
                        specular=1,
                        roughness=0.5
                    ),
                    contours={
                        "x": {"show": True, "color":"#FF9800", "width": 2},
                        "y": {"show": True, "color":"#FF9800", "width": 2},
                        "z": {"show": True, "color":"#FF9800", "width": 2}
                    },
                    hovertemplate=(
                        "<b>WACC</b>: %{x:.2f}%<br>" +
                        "<b>Growth</b>: %{y:.2f}%<br>" +
                        "<b>EV (Perpetuity)</b>: £%{z:.2f}M<br>" +
                        "<extra></extra>"
                    )
                ),
                row=1, col=2
            )

            # Add a marker for current values in second surface
            fig.add_trace(
                go.Scatter3d(
                    x=[wacc_base * 100],
                    y=[growth_base * 100],
                    z=[ev_perpetuity],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='circle'
                    ),
                    name='Current Perpetuity EV'
                ),
                row=1, col=2
            )

            # Update the layout
            fig.update_layout(
                height=600,
                scene=dict(
                    xaxis_title="WACC (%)",
                    yaxis_title="Terminal Growth (%)",
                    zaxis_title="Enterprise Value (£M)",
                    xaxis=dict(
                        gridcolor='rgba(0,0,0,0.1)',
                        showbackground=False,
                        zeroline=False
                    ),
                    yaxis=dict(
                        gridcolor='rgba(0,0,0,0.1)',
                        showbackground=False,
                        zeroline=False
                    ),
                    zaxis=dict(
                        gridcolor='rgba(0,0,0,0.1)',
                        showbackground=False,
                        zeroline=False
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2),
                    ),
                ),
                scene2=dict(
                    xaxis_title="WACC (%)",
                    yaxis_title="Terminal Growth (%)",
                    zaxis_title="Enterprise Value (£M)",
                    xaxis=dict(
                        gridcolor='rgba(0,0,0,0.1)',
                        showbackground=False,
                        zeroline=False
                    ),
                    yaxis=dict(
                        gridcolor='rgba(0,0,0,0.1)',
                        showbackground=False,
                        zeroline=False
                    ),
                    zaxis=dict(
                        gridcolor='rgba(0,0,0,0.1)',
                        showbackground=False,
                        zeroline=False
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2),
                    ),
                ),
                margin=dict(l=0, r=0, t=60, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="Reset View",
                                method="relayout",
                                args=[{
                                    "scene.camera.eye": dict(x=1.5, y=1.5, z=1.2),
                                    "scene2.camera.eye": dict(x=1.5, y=1.5, z=1.2),
                                }]
                            ),
                            dict(
                                label="Top View",
                                method="relayout",
                                args=[{
                                    "scene.camera.eye": dict(x=0, y=0, z=2.5),
                                    "scene2.camera.eye": dict(x=0, y=0, z=2.5),
                                }]
                            ),
                            dict(
                                label="Side View",
                                method="relayout",
                                args=[{
                                    "scene.camera.eye": dict(x=2.5, y=0, z=0),
                                    "scene2.camera.eye": dict(x=2.5, y=0, z=0),
                                }]
                            )
                        ],
                        direction="down",
                        pad={"r": 10, "t": 10},
                        x=0.05,
                        y=1.1,
                        xanchor="left",
                        yanchor="top"
                    ),
                ]
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add an explanatory note
            st.markdown("""
            <div style="background-color:#E3F2FD; padding:15px; border-radius:5px; margin-top:10px;">
              <h4 style="margin-top:0;">3D Visualization Insights</h4>
              <p>This interactive 3D surface shows how Enterprise Value changes with different combinations of WACC and Terminal Growth Rate.
              The red dot represents the current values.</p>
              <ul>
                <li><b>Drag</b> to rotate the 3D view</li>
                <li><b>Scroll</b> to zoom in/out</li>
                <li>Use the <b>buttons</b> at the top left to switch between different predefined views</li>
                <li><b>Hover</b> over the surface to see precise values</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)

    def display_share_price_sunburst(self):
        """Display an advanced sunburst chart for share price components"""
        st.subheader("Share Price Components - Interactive Sunburst")

        # Extract relevant values
        enterprise_value = max(self.variables['ev_perpetuity'], self.variables['ev_multiples'])
        net_debt = self.variables.get('net_debt', enterprise_value * 0.3)  # Estimated if not available
        equity_value = enterprise_value - net_debt

        # Create a hierarchical dataset for the sunburst chart
        sunburst_data = {
            'labels': ['Total Enterprise Value', 'Net Debt', 'Equity Value',
                      'Historical FCF Value', 'Terminal Value'],
            'parents': ['', 'Total Enterprise Value', 'Total Enterprise Value',
                       'Equity Value', 'Equity Value'],
            'values': [enterprise_value, net_debt, equity_value,
                      equity_value * 0.35, equity_value * 0.65],  # Approximation
            'textinfo': 'label+percent entry+value',
            'marker': {'colors': [
                '#1E88E5', '#F44336', '#4CAF50',
                '#9C27B0', '#FF9800'
            ]}
        }

        # Create interactive sunburst chart
        with chart_container(use_container_width=True):
            fig = go.Figure(go.Sunburst(
                labels=sunburst_data['labels'],
                parents=sunburst_data['parents'],
                values=sunburst_data['values'],
                branchvalues='total',
                texttemplate='<b>%{label}</b><br>£%{value:.1f}M<br>%{percentEntry:.1%}',
                hovertemplate='<b>%{label}</b><br>Value: £%{value:.2f}M<br>Percentage: %{percentEntry:.2%}<extra></extra>',
                marker=dict(
                    colors=sunburst_data['marker']['colors'],
                    line=dict(color='white', width=1)
                ),
                rotation=90
            ))

            fig.update_layout(
                height=600,
                margin=dict(t=10, l=10, r=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                uniformtext=dict(minsize=12, mode='hide'),
                # Add annotations
                annotations=[
                    dict(
                        x=0.5,
                        y=1.05,
                        xref='paper',
                        yref='paper',
                        text='Value Components (£M)',
                        showarrow=False,
                        font=dict(
                            family='Arial',
                            size=16,
                            color='rgba(0,0,0,0.8)'
                        )
                    )
                ]
            )

            st.plotly_chart(fig, use_container_width=True)

        # Add explanation cards
        col1, col2 = st.columns(2)
        with col1:
            with stylable_container(
                key="ev_explanation",
                css_styles="""
                {
                    background-color: #E3F2FD;
                    border-radius: 10px;
                    padding: 15px;
                    height: 100%;
                }
                """,
            ):
                st.markdown("### Enterprise Value")
                st.markdown("""
                Enterprise Value represents the total value of the company including debt.
                It's calculated as the present value of all future cash flows.

                **Key Components:**
                - **Equity Value**: The portion of value belonging to shareholders
                - **Net Debt**: Total debt minus cash and cash equivalents
                """)

        with col2:
            with stylable_container(
                key="share_price_explanation",
                css_styles="""
                {
                    background-color: #FFF8E1;
                    border-radius: 10px;
                    padding: 15px;
                    height: 100%;
                }
                """,
            ):
                st.markdown("### Share Price Derivation")

                # Calculate share price using a simple formula
                shares_outstanding = self.variables.get('shares_outstanding',
                                                       equity_value / self.variables['share_price_perpetuity'])
                calculated_price = equity_value / shares_outstanding

                st.markdown(f"""
                Share Price is derived from Equity Value divided by shares outstanding.

                **Current Calculation:**
                - Equity Value: £{equity_value:.2f}M
                - Shares Outstanding: {shares_outstanding:.2f}M
                - Calculated Share Price: £{calculated_price:.2f}
                """)

    def display_wacc_analysis_dashboard(self):
        """Display an advanced interactive WACC analysis dashboard"""
        st.subheader("Discount Rate (WACC) Sensitivity Dashboard")

        # Get base values
        wacc_base = self.variables['wacc']

        # Create a multi-panel dashboard
        tab1, tab2 = st.tabs(["Interactive WACC Analysis", "WACC Components"])

        with tab1:
            wacc_col1, wacc_col2 = st.columns([2, 1])

            with wacc_col1:
                # Create interactive controls
                with stylable_container(
                    key="wacc_controls",
                    css_styles="""
                    {
                        background-color: rgba(0,0,0,0.03);
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 15px;
                    }
                    """,
                ):
                    st.markdown("#### Sensitivity Parameters")

                    # Two column layout for controls
                    control_col1, control_col2 = st.columns([1, 1])

                    with control_col1:
                        # Allow user to adjust the range
                        wacc_range_pct = st.slider(
                            "WACC Sensitivity Range (±%)",
                            min_value=1.0,
                            max_value=5.0,
                            value=3.0,
                            step=0.5,
                            key="wacc_range_slider_advanced"
                        )

                    with control_col2:
                        # Allow user to adjust sensitivity model
                        wacc_sensitivity = st.select_slider(
                            "Sensitivity Model",
                            options=["Conservative", "Balanced", "Aggressive"],
                            value="Balanced"
                        )

                # Create an advanced streamlit chart with gradient
                # Define the sensitivity factors
                sensitivity_factors = {
                    "Conservative": {"multiples": 0.5, "perpetuity": 0.8},
                    "Balanced": {"multiples": 0.7, "perpetuity": 1.2},
                    "Aggressive": {"multiples": 0.9, "perpetuity": 1.5}
                }

                multiples_factor = sensitivity_factors[wacc_sensitivity]["multiples"]
                perpetuity_factor = sensitivity_factors[wacc_sensitivity]["perpetuity"]

                # Calculate WACC range with more points for smoother curves
                wacc_range = np.linspace(
                    max(0.01, wacc_base - (wacc_range_pct/100)),
                    wacc_base + (wacc_range_pct/100),
                    30  # More points for smoother curves
                )

                # Calculate effect on share prices
                wacc_effect_multiples = []
                wacc_effect_perpetuity = []

                for w in wacc_range:
                    # Effect on multiples method (less sensitive to WACC)
                    multiples_adjustment = np.power(wacc_base / w, multiples_factor)
                    adjusted_price_multiples = self.variables['share_price_multiples'] * multiples_adjustment
                    wacc_effect_multiples.append(adjusted_price_multiples)

                    # Effect on perpetuity method (more sensitive to WACC)
                    perpetuity_adjustment = np.power(wacc_base / w, perpetuity_factor)
                    adjusted_price_perpetuity = self.variables['share_price_perpetuity'] * perpetuity_adjustment
                    wacc_effect_perpetuity.append(adjusted_price_perpetuity)

                # Create data for the altair chart
                wacc_data = []
                for i, w in enumerate(wacc_range):
                    wacc_data.append({
                        'WACC': f"{w*100:.2f}%",
                        'WACC_numeric': w*100,
                        'Share Price': wacc_effect_multiples[i],
                        'Method': 'Multiples'
                    })
                    wacc_data.append({
                        'WACC': f"{w*100:.2f}%",
                        'WACC_numeric': w*100,
                        'Share Price': wacc_effect_perpetuity[i],
                        'Method': 'Perpetuity'
                    })

                wacc_df = pd.DataFrame(wacc_data)

                # Create interactive Altair chart
                wacc_highlight = alt.selection_single(
                    on='mouseover',
                    fields=['WACC'],
                    nearest=True
                )

                # Create base chart
                base = alt.Chart(wacc_df).encode(
                    x=alt.X('WACC_numeric:Q', axis=alt.Axis(title='WACC (%)', format='.2f')),
                    y=alt.Y('Share Price:Q', axis=alt.Axis(title='Share Price (£)', format='£,.2f')),
                    color=alt.Color('Method:N', scale=alt.Scale(
                        domain=['Multiples', 'Perpetuity'],
                        range=['#1E88E5', '#FF9800']
                    ))
                )

                # Create main line chart with gradient fill
                lines = base.mark_line(
                    point=True,
                    strokeWidth=3,
                    interpolate='monotone'
                ).encode(
                    opacity=alt.condition(wacc_highlight, alt.value(1), alt.value(0.7)),
                    tooltip=[
                        alt.Tooltip('WACC:N', title='WACC'),
                        alt.Tooltip('Share Price:Q', title='Share Price', format='£,.2f'),
                        alt.Tooltip('Method:N', title='Method')
                    ]
                )

                # Add vertical rule for current WACC
                rule = alt.Chart(pd.DataFrame([{'WACC': wacc_base*100}])).mark_rule(
                    color='red',
                    strokeWidth=1,
                    strokeDash=[5, 5]
                ).encode(
                    x='WACC:Q',
                    tooltip=alt.Tooltip('WACC:Q', title='Current WACC', format='.2f%')
                )

                # Add points for current values
                points = alt.Chart(pd.DataFrame([
                    {'WACC': wacc_base*100, 'Share Price': self.variables['share_price_multiples'], 'Method': 'Multiples'},
                    {'WACC': wacc_base*100, 'Share Price': self.variables['share_price_perpetuity'], 'Method': 'Perpetuity'}
                ])).mark_circle(
                    size=100,
                    opacity=0.8,
                    stroke='white',
                    strokeWidth=1
                ).encode(
                    x='WACC:Q',
                    y='Share Price:Q',
                    color='Method:N',
                    tooltip=[
                        alt.Tooltip('WACC:Q', title='Current WACC', format='.2f%'),
                        alt.Tooltip('Share Price:Q', title='Current Share Price', format='£,.2f'),
                        alt.Tooltip('Method:N', title='Method')
                    ]
                )

                # Combine all layers
                chart = (lines + rule + points).add_selection(
                    wacc_highlight
                ).properties(
                    width=600,
                    height=400
                ).configure_axisX(
                    grid=True,
                    gridOpacity=0.2
                ).configure_axisY(
                    grid=True,
                    gridOpacity=0.2
                ).configure_view(
                    strokeOpacity=0
                )

                st.altair_chart(chart, use_container_width=True)

            with wacc_col2:
                # Create interactive WACC calculator
                with stylable_container(
                    key="wacc_calculator",
                    css_styles="""
                    {
                        background-color: #F3E5F5;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 15px;
                    }
                    """,
                ):
                    st.markdown("#### WACC Calculator")

                    # Get current WACC components (estimated if not available)
                    risk_free_rate = self.variables.get('risk_free_rate', 0.035)
                    equity_risk_premium = self.variables.get('market_risk_premium', 0.05)
                    beta = self.variables.get('beta', 1.2)
                    cost_of_debt = self.variables.get('cost_of_debt', 0.045)
                    tax_rate = self.variables.get('tax_rate', 0.19)
                    debt_weight = self.variables.get('debt_weight', 0.3)
                    equity_weight = 1 - debt_weight

                    # Allow user to adjust components
                    adjusted_rfr = st.slider(
                        "Risk-Free Rate (%)",
                        min_value=1.0,
                        max_value=6.0,
                        value=float(risk_free_rate * 100),
                        step=0.1
                    ) / 100

                    adjusted_erp = st.slider(
                        "Equity Risk Premium (%)",
                        min_value=2.0,
                        max_value=10.0,
                        value=float(equity_risk_premium * 100),
                        step=0.1
                    ) / 100

                    adjusted_beta = st.slider(
                        "Beta",
                        min_value=0.5,
                        max_value=2.0,
                        value=float(beta),
                        step=0.05
                    )

                    adjusted_cod = st.slider(
                        "Cost of Debt (%)",
                        min_value=1.0,
                        max_value=8.0,
                        value=float(cost_of_debt * 100),
                        step=0.1
                    ) / 100

                    adjusted_debt = st.slider(
                        "Debt Weight (%)",
                        min_value=0.0,
                        max_value=70.0,
                        value=float(debt_weight * 100),
                        step=1.0
                    ) / 100

                    adjusted_equity = 1 - adjusted_debt

                    # Calculate adjusted WACC
                    cost_of_equity = adjusted_rfr + adjusted_beta * adjusted_erp
                    after_tax_cod = adjusted_cod * (1 - tax_rate)
                    adjusted_wacc = (cost_of_equity * adjusted_equity) + (after_tax_cod * adjusted_debt)

                    # Display calculated WACC
                    st.markdown(f"#### Calculated WACC: {adjusted_wacc*100:.2f}%")

                    # Display difference from current WACC
                    wacc_diff = adjusted_wacc - wacc_base
                    wacc_pct_diff = (wacc_diff / wacc_base) * 100

                    if abs(wacc_diff) > 0.0001:
                        direction = "higher" if wacc_diff > 0 else "lower"
                        color = "red" if wacc_diff > 0 else "green"
                        st.markdown(f"<span style='color:{color};font-weight:500;'>This is {abs(wacc_pct_diff):.2f}% {direction} than current WACC ({wacc_base*100:.2f}%)</span>", unsafe_allow_html=True)

                    # Estimate impact on share price
                    price_impact = ((wacc_base / adjusted_wacc) ** perpetuity_factor - 1) * 100
                    impact_direction = "increase" if price_impact > 0 else "decrease"
                    impact_color = "green" if price_impact > 0 else "red"

                    st.markdown(f"<div style='background-color:white;padding:10px;border-radius:5px;margin-top:15px;'>Estimated <span style='color:{impact_color};font-weight:500;'>{impact_direction}</span> in share price: <span style='color:{impact_color};font-weight:500;'>{abs(price_impact):.2f}%</span></div>", unsafe_allow_html=True)

                # Add explanatory card
                with stylable_container(
                    key="wacc_explanation",
                    css_styles="""
                    {
                        background-color: rgba(0,0,0,0.03);
                        border-radius: 10px;
                        padding: 15px;
                        margin-top: 15px;
                    }
                    """,
                ):
                    st.markdown("#### WACC Impact on Valuation")
                    st.markdown("""
                    The Weighted Average Cost of Capital (WACC) serves as the discount rate in DCF modeling.

                    **Key impacts:**
                    - A lower WACC results in higher valuation
                    - A higher WACC results in lower valuation
                    - The Perpetuity method is more sensitive to WACC changes

                    WACC is calculated by weighting the cost of equity and after-tax cost of debt by their proportions in the company's capital structure.
                    """)

        with tab2:
            # Create WACC waterfall chart showing components
            cost_of_equity = risk_free_rate + beta * equity_risk_premium
            after_tax_cod = cost_of_debt * (1 - tax_rate)
            equity_contribution = cost_of_equity * equity_weight
            debt_contribution = after_tax_cod * debt_weight

            # Create data for WACC component visualization
            fig = go.Figure()

            # Add waterfall chart showing WACC build-up
            fig.add_trace(go.Waterfall(
                name="WACC Components",
                orientation="v",
                measure=["relative", "relative", "total"],
                x=["Equity Component", "Debt Component (After-Tax)", "Total WACC"],
                textposition="outside",
                text=[
                    f"{equity_contribution*100:.2f}%",
                    f"{debt_contribution*100:.2f}%",
                    f"{wacc_base*100:.2f}%"
                ],
                y=[equity_contribution*100, debt_contribution*100, 0],
                connector={"line": {"color": "rgba(0,0,0,0.3)"}},
                decreasing={"marker": {"color": "#1E88E5"}},
                increasing={"marker": {"color": "#FF9800"}},
                totals={"marker": {"color": "#4CAF50"}},
                hoverinfo="text",
                hovertext=[
                    f"Equity Component: {equity_contribution*100:.2f}%<br>" +
                    f"(CoE: {cost_of_equity*100:.2f}% × Weight: {equity_weight*100:.0f}%)",

                    f"Debt Component: {debt_contribution*100:.2f}%<br>" +
                    f"(After-tax CoD: {after_tax_cod*100:.2f}% × Weight: {debt_weight*100:.0f}%)",

                    f"Total WACC: {wacc_base*100:.2f}%"
                ]
            ))

            # Update layout with better formatting
            fig.update_layout(
                title={
                    'text': "WACC Build-up Analysis",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(
                        family="Arial",
                        size=20,
                        color="rgba(0, 0, 0, 0.85)"
                    )
                },
                yaxis_title="Percentage (%)",
                height=500,
                template="plotly_white",
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=0.5,
                    gridcolor='rgba(0,0,0,0.1)',
                    zeroline=True,
                    zerolinecolor='rgba(0,0,0,0.2)',
                    zerolinewidth=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add component details in a structured table
            col1, col2 = st.columns(2)

            with col1:
                with stylable_container(
                    key="coe_details",
                    css_styles="""
                    {
                        background-color: #E3F2FD;
                        border-radius: 10px;
                        padding: 15px;
                    }
                    """,
                ):
                    st.markdown("#### Cost of Equity Details")
                    st.markdown(f"""
                    **Formula:** Risk-Free Rate + (Beta × Equity Risk Premium)

                    **Components:**
                    - Risk-Free Rate: {risk_free_rate*100:.2f}%
                    - Company Beta: {beta:.2f}
                    - Equity Risk Premium: {equity_risk_premium*100:.2f}%

                    **Calculation:**
                    {risk_free_rate*100:.2f}% + ({beta:.2f} × {equity_risk_premium*100:.2f}%) = {cost_of_equity*100:.2f}%

                    **Weighted Contribution:**
                    {cost_of_equity*100:.2f}% × {equity_weight*100:.0f}% = {equity_contribution*100:.2f}%
                    """)

            with col2:
                with stylable_container(
                    key="cod_details",
                    css_styles="""
                    {
                        background-color: #FFF8E1;
                        border-radius: 10px;
                        padding: 15px;
                    }
                    """,
                ):
                    st.markdown("#### Cost of Debt Details")
                    st.markdown(f"""
                    **Formula:** Cost of Debt × (1 - Tax Rate)

                    **Components:**
                    - Pre-tax Cost of Debt: {cost_of_debt*100:.2f}%
                    - Corporate Tax Rate: {tax_rate*100:.0f}%

                    **Calculation:**
                    {cost_of_debt*100:.2f}% × (1 - {tax_rate:.2f}) = {after_tax_cod*100:.2f}%

                    **Weighted Contribution:**
                    {after_tax_cod*100:.2f}% × {debt_weight*100:.0f}% = {debt_contribution*100:.2f}%
                    """)

    def display_two_factor_heatmap(self):
        """Display an advanced 2-factor sensitivity heatmap"""
        st.subheader("Two-Factor Sensitivity Analysis - Interactive Heatmap")

        # Allow user to select factors to analyze
        col1, col2 = st.columns(2)

        with col1:
            factor1 = st.selectbox(
                "First Factor",
                ["Discount Rate (WACC)", "Terminal Growth Rate", "Revenue Growth", "Operating Margin"],
                key="factor1_select"
            )

        with col2:
            factor2 = st.selectbox(
                "Second Factor",
                ["Terminal Growth Rate", "Discount Rate (WACC)", "Revenue Growth", "Operating Margin"],
                key="factor2_select",
                index=1
            )

        # Map selected factors to variables and ranges
        factor_mapping = {
            "Discount Rate (WACC)": {
                "variable": "wacc",
                "range_pct": [-25, 25],
                "step_pct": 5,
                "format": "%"
            },
            "Terminal Growth Rate": {
                "variable": "terminal_growth",
                "range_pct": [-50, 50],
                "step_pct": 10,
                "format": "%"
            },
            "Revenue Growth": {
                "variable": "revenue_growth",
                "range_pct": [-30, 30],
                "step_pct": 6,
                "format": "%"
            },
            "Operating Margin": {
                "variable": "operating_margin",
                "range_pct": [-20, 20],
                "step_pct": 4,
                "format": "%"
            }
        }

        # Extract factor details
        factor1_details = factor_mapping[factor1]
        factor2_details = factor_mapping[factor2]

        # Get base values
        factor1_base = self.variables.get(factor1_details["variable"], 0.05)
        factor2_base = self.variables.get(factor2_details["variable"], 0.02)
        base_share_price = self.variables["share_price_perpetuity"]

        # Create arrays for percentage changes from base
        factor1_pcts = np.arange(
            factor1_details["range_pct"][0],
            factor1_details["range_pct"][1] + factor1_details["step_pct"],
            factor1_details["step_pct"]
        )

        factor2_pcts = np.arange(
            factor2_details["range_pct"][0],
            factor2_details["range_pct"][1] + factor2_details["step_pct"],
            factor2_details["step_pct"]
        )

        # Calculate absolute values for each factor
        factor1_values = []
        for pct in factor1_pcts:
            factor1_values.append(factor1_base * (1 + pct/100))

        factor2_values = []
        for pct in factor2_pcts:
            factor2_values.append(factor2_base * (1 + pct/100))

        # Create labels for display
        factor1_labels = []
        for val in factor1_values:
            if factor1_details["format"] == "%":
                factor1_labels.append(f"{val*100:.2f}%")
            else:
                factor1_labels.append(f"{val:.2f}")

        factor2_labels = []
        for val in factor2_values:
            if factor2_details["format"] == "%":
                factor2_labels.append(f"{val*100:.2f}%")
            else:
                factor2_labels.append(f"{val:.2f}")

        # Calculate share prices for each combination
        share_prices = []
        for f1 in factor1_values:
            row = []
            for f2 in factor2_values:
                # Create a simplified model to estimate share price impact
                if factor1 == "Discount Rate (WACC)" and factor2 == "Terminal Growth Rate":
                    # Special case for WACC and growth rate - use Gordon Growth formula
                    term_value_factor = (1 + f2) / (f1 - f2)
                    base_term_factor = (1 + factor2_base) / (factor1_base - factor2_base)
                    price = base_share_price * (term_value_factor / base_term_factor)
                else:
                    # Simplified approximation for other combinations
                    f1_impact = 0
                    if factor1 == "Discount Rate (WACC)":
                        f1_impact = -5 * (f1 - factor1_base) / factor1_base
                    elif factor1 == "Terminal Growth Rate":
                        f1_impact = 3 * (f1 - factor1_base) / factor1_base
                    elif factor1 == "Revenue Growth":
                        f1_impact = 1.5 * (f1 - factor1_base) / factor1_base
                    elif factor1 == "Operating Margin":
                        f1_impact = 2 * (f1 - factor1_base) / factor1_base

                    f2_impact = 0
                    if factor2 == "Discount Rate (WACC)":
                        f2_impact = -5 * (f2 - factor2_base) / factor2_base
                    elif factor2 == "Terminal Growth Rate":
                        f2_impact = 3 * (f2 - factor2_base) / factor2_base
                    elif factor2 == "Revenue Growth":
                        f2_impact = 1.5 * (f2 - factor2_base) / factor2_base
                    elif factor2 == "Operating Margin":
                        f2_impact = 2 * (f2 - factor2_base) / factor2_base

                    price = base_share_price * (1 + f1_impact + f2_impact)

                row.append(price)
            share_prices.append(row)

        # Convert to numpy array for better processing
        z_data = np.array(share_prices)

        # Calculate percentage changes from base
        base_price_idx = (len(factor1_pcts) // 2, len(factor2_pcts) // 2)
        pct_changes = np.zeros_like(z_data)

        for i in range(z_data.shape[0]):
            for j in range(z_data.shape[1]):
                pct_changes[i, j] = ((z_data[i, j] / base_share_price) - 1) * 100

        # Create heatmap with advanced styling
        fig = go.Figure()

        # Add heatmap trace
        fig.add_trace(go.Heatmap(
            z=pct_changes,
            x=factor2_labels,
            y=factor1_labels,
            colorscale=[
                [0, '#d73027'],            # Dark red (lowest)
                [0.25, '#f46d43'],         # Red-orange
                [0.35, '#fdae61'],         # Light orange
                [0.45, '#fee090'],         # Pale yellow
                [0.5, '#ffffbf'],          # Light yellow (middle)
                [0.55, '#e0f3f8'],         # Pale blue
                [0.65, '#abd9e9'],         # Light blue
                [0.75, '#74add1'],         # Blue
                [1, '#4575b4']             # Dark blue (highest)
            ],
            zmid=0,  # Center the color scale at 0
            hovertemplate=(
                f"<b>{factor1}</b>: %{{y}}<br>" +
                f"<b>{factor2}</b>: %{{x}}<br>" +
                "<b>Share Price</b>: £%{customdata:.2f}<br>" +
                "<b>% Change</b>: %{z:.1f}%<br>" +
                "<extra></extra>"
            ),
            customdata=z_data,  # Include absolute values for hover
            colorbar=dict(
                title="% Change<br>in Share Price",
                titleside="top",
                tickformat=".0f",
                ticksuffix="%",
                ticks="outside"
            )
        ))

        # Add marker for base case
        mid_x = len(factor2_labels) // 2
        mid_y = len(factor1_labels) // 2

        # Add annotation to mark the base case
        fig.add_annotation(
            x=factor2_labels[mid_x],
            y=factor1_labels[mid_y],
            text="Base<br>Case",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#000000",
            ax=0,
            ay=-40,
            font=dict(
                family="Arial",
                size=12,
                color="#000000"
            ),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#000000",
            borderwidth=1,
            borderpad=4,
            align="center"
        )

        # Enhance the layout
        fig.update_layout(
            title={
                'text': f"Share Price Sensitivity to {factor1} and {factor2}",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            width=800,
            height=700,
            xaxis_title=factor2,
            yaxis_title=factor1,
            xaxis=dict(
                tickangle=-45,
                side="bottom",
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                tickfont=dict(size=10)
            ),
            margin=dict(l=50, r=50, t=100, b=100)
        )

        # Add contour lines to highlight regions
        fig.add_trace(go.Contour(
            z=pct_changes,
            x=factor2_labels,
            y=factor1_labels,
            showscale=False,
            contours=dict(
                coloring='none',
                showlabels=True,
                start=-30,
                end=30,
                size=10,
                labelfont=dict(
                    family="Arial",
                    size=10,
                    color="black"
                )
            ),
            line=dict(
                width=1,
                color='rgba(0,0,0,0.5)'
            )
        ))

        # Add a custom shape to highlight optimal region
        max_idx = np.unravel_index(np.argmax(pct_changes), pct_changes.shape)
        max_x = factor2_labels[max_idx[1]]
        max_y = factor1_labels[max_idx[0]]
        max_pct = pct_changes[max_idx]

        fig.add_trace(go.Scatter(
            x=[max_x],
            y=[max_y],
            mode='markers',
            marker=dict(
                size=14,
                color='rgba(0,0,0,0)',
                line=dict(
                    color='black',
                    width=2
                ),
                symbol='circle-open'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_annotation(
            x=max_x,
            y=max_y,
            text=f"Optimal<br>+{max_pct:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#00AA00",
            ax=40,
            ay=40,
            font=dict(
                family="Arial",
                size=12,
                color="#00AA00"
            ),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#00AA00",
            borderwidth=1,
            borderpad=4,
            align="center"
        )

        # Add interactivity with buttons for different views
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=[
                        dict(
                            args=[{'reversescale': False}],
                            label="Default Colorscale",
                            method="restyle"
                        ),
                        dict(
                            args=[{'reversescale': True}],
                            label="Reverse Colorscale",
                            method="restyle"
                        ),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add explanatory text
        st.markdown("""
        <div style="background-color:rgba(0,0,0,0.03); padding:15px; border-radius:5px; margin-top:10px;">
          <h4 style="margin-top:0;">How to Interpret the Heatmap</h4>
          <p>This interactive heatmap shows how the share price responds to simultaneous changes in two key factors.</p>
          <ul>
            <li><b>Color intensity</b> indicates the percentage change in share price from the base case.</li>
            <li><b>Contour lines</b> connect points with the same percentage change, creating "elevation" regions.</li>
            <li>The <b>base case</b> (current parameters) is marked in the center.</li>
            <li>The <b>optimal point</b> shows parameter combinations yielding the highest share price.</li>
          </ul>
          <p>Hover over the heatmap to see precise values and percentage changes.</p>
        </div>
        """, unsafe_allow_html=True)

    def display_visual_dashboard(self):
        """Display the full advanced visual dashboard"""

        # Create header with key metrics
        self.display_header_dashboard()

        # Create tabs for different visualization types
        tab1, tab2, tab3, tab4 = st.tabs([
            "Share Price Analysis",
            "Enterprise Value Analysis",
            "Sensitivity Analysis",
            "Advanced Visualizations"
        ])

        with tab1:
            # Display share price waterfall chart - maintained from original
            self.dcf.display_share_price_chart()

            # Add sunburst diagram for share price components
            self.display_share_price_sunburst()

        with tab2:
            # Display original enterprise value chart for comparison
            self.dcf.display_enterprise_value_chart()

            # Add 3D enterprise value surface
            self.display_enterprise_value_3d()

        with tab3:
            # Display advanced WACC analysis
            self.display_wacc_analysis_dashboard()

            # Display two-factor sensitivity heatmap
            self.display_two_factor_heatmap()

        with tab4:
            # Offer additional dashboards and visualizations
            st.info("Select an advanced visualization to explore:")

            adv_option = st.radio(
                "Advanced Visualization Type",
                ["Scenario Comparisons", "Monte Carlo Simulation", "Value Drivers Analysis"],
                horizontal=True
            )

            if adv_option == "Scenario Comparisons":
                # Create scenario comparison visualization
                # This is simplified for the example
                st.subheader("Scenario Analysis - Share Price Impact")

                # Define scenarios
                scenarios = {
                    "Base Case": {
                        "wacc": self.variables['wacc'],
                        "growth": self.variables['terminal_growth'],
                        "rev_growth": 0.03,
                        "margin": 0.10
                    },
                    "Bull Case": {
                        "wacc": self.variables['wacc'] * 0.9,
                        "growth": self.variables['terminal_growth'] * 1.2,
                        "rev_growth": 0.05,
                        "margin": 0.12
                    },
                    "Bear Case": {
                        "wacc": self.variables['wacc'] * 1.1,
                        "growth": self.variables['terminal_growth'] * 0.8,
                        "rev_growth": 0.01,
                        "margin": 0.08
                    },
                    "Recession": {
                        "wacc": self.variables['wacc'] * 1.2,
                        "growth": self.variables['terminal_growth'] * 0.5,
                        "rev_growth": -0.02,
                        "margin": 0.05
                    },
                    "Expansion": {
                        "wacc": self.variables['wacc'] * 0.85,
                        "growth": self.variables['terminal_growth'] * 1.3,
                        "rev_growth": 0.06,
                        "margin": 0.14
                    }
                }

                # Calculate share prices for each scenario
                share_prices = {}
                for scenario, params in scenarios.items():
                    # Simple calculation for demonstration
                    growth_factor = params["growth"] / self.variables['terminal_growth']
                    wacc_factor = self.variables['wacc'] / params["wacc"]
                    rev_factor = (1 + params["rev_growth"]) / (1 + 0.03)  # Assume 3% base case
                    margin_factor = params["margin"] / 0.10  # Assume 10% base case

                    # Weight the factors
                    combined_factor = (wacc_factor ** 1.2) * (growth_factor ** 0.8) * (rev_factor ** 0.5) * (margin_factor ** 0.7)
                    share_prices[scenario] = self.variables['share_price_perpetuity'] * combined_factor

                # Create a chart comparing scenarios
                scenarios_list = list(scenarios.keys())
                prices_list = [share_prices[s] for s in scenarios_list]

                # Calculate percentage changes
                base_price = share_prices["Base Case"]
                pct_changes = [(p / base_price - 1) * 100 for p in prices_list]

                # Color map based on percentage change
                colors = []
                for pct in pct_changes:
                    if pct > 20:
                        colors.append('#1B5E20')  # Dark green
                    elif pct > 0:
                        colors.append('#4CAF50')  # Green
                    elif pct > -20:
                        colors.append('#F44336')  # Red
                    else:
                        colors.append('#B71C1C')  # Dark red

                # Create the figure
                fig = go.Figure()

                # Add bar chart
                fig.add_trace(go.Bar(
                    x=scenarios_list,
                    y=prices_list,
                    marker_color=colors,
                    text=[f"£{p:.2f}<br>({c:+.1f}%)" for p, c in zip(prices_list, pct_changes)],
                    textposition="auto",
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Share Price: £%{y:.2f}<br>" +
                        "Change: %{customdata:+.1f}%<br>" +
                        "<extra></extra>"
                    ),
                    customdata=pct_changes
                ))

                # Add line for current share price
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(scenarios_list) - 0.5,
                    y0=self.variables['current_share_price'],
                    y1=self.variables['current_share_price'],
                    line=dict(
                        color="black",
                        width=2,
                        dash="dash"
                    )
                )

                # Add annotation for current price
                fig.add_annotation(
                    x=len(scenarios_list) - 1,
                    y=self.variables['current_share_price'],
                    text=f"Current: £{self.variables['current_share_price']:.2f}",
                    showarrow=False,
                    yshift=10,
                    font=dict(
                        family="Arial",
                        size=12,
                        color="black"
                    ),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                )

                # Enhanced layout
                fig.update_layout(
                    title={
                        'text': "Share Price Across Scenarios",
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    xaxis_title="Scenario",
                    yaxis_title="Share Price (£)",
                    yaxis=dict(zeroline=False),
                    height=500,
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=80, b=40),
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show scenario details in a table
                st.subheader("Scenario Details")

                # Create a DataFrame with scenario details
                scenario_data = []
                for scenario, params in scenarios.items():
                    scenario_data.append({
                        "Scenario": scenario,
                        "WACC": f"{params['wacc']*100:.2f}%",
                        "Terminal Growth": f"{params['growth']*100:.2f}%",
                        "Revenue Growth": f"{params['rev_growth']*100:.2f}%",
                        "Operating Margin": f"{params['margin']*100:.2f}%",
                        "Share Price": f"£{share_prices[scenario]:.2f}",
                        "% Change": f"{pct_changes[scenarios_list.index(scenario)]:+.1f}%"
                    })

                scenario_df = pd.DataFrame(scenario_data)
                st.table(scenario_df)

            elif adv_option == "Monte Carlo Simulation":
                # Create a Monte Carlo simulation visualization
                st.subheader("Monte Carlo Simulation - Share Price Distribution")

                # Set number of simulations
                n_simulations = 1000

                # Get base values
                base_wacc = self.variables['wacc']
                base_growth = self.variables['terminal_growth']
                base_price = self.variables['share_price_perpetuity']

                # Create random variations of parameters
                np.random.seed(42)  # For reproducibility

                # Monte Carlo simulation parameters
                wacc_samples = np.random.normal(base_wacc, base_wacc * 0.10, n_simulations)  # 10% standard deviation
                growth_samples = np.random.normal(base_growth, base_growth * 0.15, n_simulations)  # 15% standard deviation
                revenue_samples = np.random.normal(0.03, 0.01, n_simulations)  # Mean 3%, SD 1%
                margin_samples = np.random.normal(0.10, 0.015, n_simulations)  # Mean 10%, SD 1.5%

                # Ensure reasonable bounds
                wacc_samples = np.clip(wacc_samples, base_wacc * 0.7, base_wacc * 1.3)
                growth_samples = np.clip(growth_samples, base_growth * 0.5, base_growth * 1.5)
                revenue_samples = np.clip(revenue_samples, -0.02, 0.08)
                margin_samples = np.clip(margin_samples, 0.05, 0.15)

                # Calculate share prices
                share_prices = []
                for i in range(n_simulations):
                    # Calculate using Gordon Growth Model with random factors
                    growth_factor = growth_samples[i] / base_growth
                    wacc_factor = base_wacc / wacc_samples[i]
                    rev_factor = (1 + revenue_samples[i]) / (1 + 0.03)
                    margin_factor = margin_samples[i] / 0.10

                    # Weight the factors
                    combined_factor = (wacc_factor ** 1.2) * (growth_factor ** 0.8) * (rev_factor ** 0.5) * (margin_factor ** 0.7)
                    share_prices.append(base_price * combined_factor)

                # Create the visualization
                fig = go.Figure()

                # Add histogram
                fig.add_trace(go.Histogram(
                    x=share_prices,
                    nbinsx=30,
                    marker_color='rgba(33, 150, 243, 0.7)',
                    marker_line=dict(
                        color='rgba(33, 150, 243, 1)',
                        width=1
                    ),
                    hovertemplate=(
                        "Share Price Range: £%{x:.2f}<br>" +
                        "Frequency: %{y}<br>" +
                        "<extra></extra>"
                    )
                ))

                # Calculate percentiles for analysis
                p10 = np.percentile(share_prices, 10)
                p25 = np.percentile(share_prices, 25)
                p50 = np.percentile(share_prices, 50)
                p75 = np.percentile(share_prices, 75)
                p90 = np.percentile(share_prices, 90)

                # Add vertical lines for key values
                fig.add_vline(
                    x=self.variables['current_share_price'],
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Current: £{self.variables['current_share_price']:.2f}",
                    annotation_position="top right"
                )

                fig.add_vline(
                    x=p50,
                    line_width=2,
                    line_color="green",
                    annotation_text=f"Median: £{p50:.2f}",
                    annotation_position="top right"
                )

                # Add annotations for percentiles
                fig.add_vrect(
                    x0=p25, x1=p75,
                    fillcolor="rgba(0, 200, 0, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text="50% Confidence Interval",
                    annotation_position="bottom right",
                    annotation=dict(
                        font_size=10,
                        font_color="green"
                    )
                )

                # Enhanced layout
                fig.update_layout(
                    title={
                        'text': f"Share Price Distribution ({n_simulations:,} Simulations)",
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    xaxis_title="Share Price (£)",
                    yaxis_title="Frequency",
                    height=500,
                    template="plotly_white",
                    bargap=0.1,
                    margin=dict(l=40, r=40, t=80, b=40),
                )

                st.plotly_chart(fig, use_container_width=True)

                # Add percentile information
                st.subheader("Simulation Results")

                col1, col2 = st.columns(2)

                with col1:
                    with stylable_container(
                        key="mc_percentiles",
                        css_styles="""
                        {
                            background-color: #E3F2FD;
                            border-radius: 10px;
                            padding: 15px;
                        }
                        """,
                    ):
                        st.markdown("#### Percentile Analysis")
                        st.markdown(f"""
                        **10th Percentile:** £{p10:.2f}
                        **25th Percentile:** £{p25:.2f}
                        **50th Percentile (Median):** £{p50:.2f}
                        **75th Percentile:** £{p75:.2f}
                        **90th Percentile:** £{p90:.2f}
                        """)

                with col2:
                    with stylable_container(
                        key="mc_insights",
                        css_styles="""
                        {
                            background-color: #FFF8E1;
                            border-radius: 10px;
                            padding: 15px;
                        }
                        """,
                    ):
                        st.markdown("#### Insight Summary")

                        # Calculate probability of price being above current
                        prob_above_current = (np.array(share_prices) > self.variables['current_share_price']).mean() * 100

                        # Calculate probability of price being within ±10% of base case
                        lower_bound = base_price * 0.9
                        upper_bound = base_price * 1.1
                        prob_within_range = ((np.array(share_prices) >= lower_bound) &
                                             (np.array(share_prices) <= upper_bound)).mean() * 100

                        st.markdown(f"""
                        **Probability Above Current Price:** {prob_above_current:.1f}%
                        **Probability Within ±10% of Base Case:** {prob_within_range:.1f}%
                        **Value at Risk (10th Percentile):** {((p10 / base_price) - 1) * 100:.1f}%
                        **Upside Potential (90th Percentile):** {((p90 / base_price) - 1) * 100:.1f}%
                        """)

                st.markdown("""
                <div style="background-color:rgba(0,0,0,0.03); padding:15px; border-radius:5px; margin-top:15px;">
                  <h4 style="margin-top:0;">Monte Carlo Simulation Overview</h4>
                  <p>This simulation runs 1,000 iterations with random variations of key model inputs:</p>
                  <ul>
                    <li>WACC: varies with 10% standard deviation</li>
                    <li>Terminal Growth: varies with 15% standard deviation</li>
                    <li>Revenue Growth: mean 3%, standard deviation 1%</li>
                    <li>Operating Margin: mean 10%, standard deviation 1.5%</li>
                  </ul>
                  <p>The resulting distribution shows the range of possible share prices and their relative probabilities.</p>
                </div>
                """, unsafe_allow_html=True)

            elif adv_option == "Value Drivers Analysis":
                # Create a value drivers analysis visualization
                st.subheader("Value Drivers Analysis - Tornado Chart")

                # Define base case
                base_price = self.variables['share_price_perpetuity']

                # Define parameters and their ranges
                parameters = [
                    "Discount Rate (WACC)",
                    "Terminal Growth",
                    "Revenue Growth",
                    "Operating Margin",
                    "Tax Rate",
                    "CAPEX % of Revenue",
                    "Working Capital % of Revenue",
                ]

                # Define sensitivity percentages (negative and positive)
                sensitivity_pct = 15

                # Calculate impact on share price (simulated for example)
                # These would normally be calculated based on actual model
                negative_impacts = [
                    -18.5,  # WACC (+15%)
                    -12.3,  # Terminal Growth (-15%)
                    -7.8,   # Revenue Growth (-15%)
                    -9.5,   # Operating Margin (-15%)
                    -3.2,   # Tax Rate (+15%)
                    -4.7,   # CAPEX (+15%)
                    -2.1,   # Working Capital (+15%)
                ]

                positive_impacts = [
                    24.2,   # WACC (-15%)
                    15.8,   # Terminal Growth (+15%)
                    8.4,    # Revenue Growth (+15%)
                    10.1,   # Operating Margin (+15%)
                    3.5,    # Tax Rate (-15%)
                    5.1,    # CAPEX (-15%)
                    2.4,    # Working Capital (-15%)
                ]

                # Sort parameters by absolute impact
                abs_impacts = [max(abs(n), abs(p)) for n, p in zip(negative_impacts, positive_impacts)]
                sorted_indices = np.argsort(abs_impacts)[::-1]  # Descending order

                sorted_parameters = [parameters[i] for i in sorted_indices]
                sorted_negative = [negative_impacts[i] for i in sorted_indices]
                sorted_positive = [positive_impacts[i] for i in sorted_indices]

                # Create a tornado chart
                fig = go.Figure()

                # Add bars for negative impacts
                fig.add_trace(go.Bar(
                    y=sorted_parameters,
                    x=sorted_negative,
                    orientation='h',
                    name=f"-{sensitivity_pct}% Change",
                    marker_color='rgba(244, 67, 54, 0.7)',
                    marker_line=dict(
                        color='rgba(244, 67, 54, 1)',
                        width=1
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>" +
                        f"-{sensitivity_pct}% Change<br>" +
                        "Impact on Share Price: %{x:+.1f}%<br>" +
                        "<extra></extra>"
                    )
                ))

                # Add bars for positive impacts
                fig.add_trace(go.Bar(
                    y=sorted_parameters,
                    x=sorted_positive,
                    orientation='h',
                    name=f"+{sensitivity_pct}% Change",
                    marker_color='rgba(76, 175, 80, 0.7)',
                    marker_line=dict(
                        color='rgba(76, 175, 80, 1)',
                        width=1
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>" +
                        f"+{sensitivity_pct}% Change<br>" +
                        "Impact on Share Price: %{x:+.1f}%<br>" +
                        "<extra></extra>"
                    )
                ))

                # Add vertical line at 0
                fig.add_vline(
                    x=0,
                    line_width=1,
                    line_color="black"
                )

                # Enhanced layout
                fig.update_layout(
                    title={
                        'text': "Tornado Chart - Value Drivers Impact",
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    xaxis_title="Impact on Share Price (%)",
                    height=500,
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=40, r=40, t=80, b=40),
                    barmode='overlay',
                    bargap=0.2,
                    xaxis=dict(
                        zeroline=True,
                        zerolinecolor='black',
                        zerolinewidth=1
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                # Add explanation
                st.markdown("""
                <div style="background-color:rgba(0,0,0,0.03); padding:15px; border-radius:5px; margin-top:10px;">
                  <h4 style="margin-top:0;">Tornado Chart Interpretation</h4>
                  <p>The tornado chart ranks value drivers by their impact on share price when they are varied by ±15% from the base case.</p>
                  <ul>
                    <li><span style="color:#F44336;">Red bars</span> show the impact of adverse changes (-15% for positive drivers, +15% for negative drivers)</li>
                    <li><span style="color:#4CAF50;">Green bars</span> show the impact of favorable changes (+15% for positive drivers, -15% for negative drivers)</li>
                  </ul>
                  <p>The parameters are sorted by their absolute impact, with the most influential drivers at the top.</p>
                </div>
                """, unsafe_allow_html=True)

                # Calculate elasticity for each parameter
                st.subheader("Value Driver Elasticities")

                elasticities = []
                for i, param in enumerate(parameters):
                    # Average absolute impact divided by the change percentage
                    avg_impact = (abs(negative_impacts[i]) + abs(positive_impacts[i])) / 2
                    elasticity = avg_impact / sensitivity_pct
                    elasticities.append(elasticity)

                # Sort by elasticity
                sorted_elasticity_indices = np.argsort(elasticities)[::-1]  # Descending order
                sorted_elasticity_params = [parameters[i] for i in sorted_elasticity_indices]
                sorted_elasticities = [elasticities[i] for i in sorted_elasticity_indices]

                # Create elasticity bars
                fig_elasticity = go.Figure()

                # Add bars for elasticities
                fig_elasticity.add_trace(go.Bar(
                    x=sorted_elasticity_params,
                    y=sorted_elasticities,
                    marker_color=[
                        'rgba(33, 150, 243, 0.8)',
                        'rgba(33, 150, 243, 0.7)',
                        'rgba(33, 150, 243, 0.6)',
                        'rgba(33, 150, 243, 0.5)',
                        'rgba(33, 150, 243, 0.4)',
                        'rgba(33, 150, 243, 0.3)',
                        'rgba(33, 150, 243, 0.2)',
                    ],
                    marker_line=dict(
                        color='rgba(33, 150, 243, 1)',
                        width=1
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Elasticity: %{y:.2f}<br>" +
                        "<extra></extra>"
                    )
                ))

                # Enhanced layout
                fig_elasticity.update_layout(
                    title={
                        'text': "Value Driver Elasticities",
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    yaxis_title="Elasticity",
                    height=400,
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=80, b=100),
                    xaxis=dict(
                        tickangle=-45
                    )
                )

                st.plotly_chart(fig_elasticity, use_container_width=True)

                st.markdown("""
                <div style="background-color:#E3F2FD; padding:15px; border-radius:5px; margin-top:10px;">
                  <h4 style="margin-top:0;">Elasticity Explanation</h4>
                  <p>Elasticity measures how sensitive the share price is to changes in each parameter, calculated as:</p>
                  <p style="text-align:center;font-style:italic;">Elasticity = % Change in Share Price / % Change in Parameter</p>
                  <p>Higher elasticity means the parameter has greater influence on valuation. Parameters with elasticity > 1.0 have a disproportionate impact on share price.</p>
                </div>
                """, unsafe_allow_html=True)
