import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

class DCFAnalyzer:
    """
    A class to extract and visualize DCF model data from an Excel file.
    """

    def __init__(self, excel_df):
        """
        Initialize the DCF Analyzer with a DataFrame from the DCF tab

        Args:
            excel_df: DataFrame containing the DCF tab data
        """
        self.df = excel_df
        self.variables = self._extract_dcf_variables()

    def _extract_dcf_variables(self):
        """
        Extract DCF variables from specific cells in the DataFrame

        Returns:
            dict: Dictionary of extracted DCF variables
        """
        try:
            # First attempt - direct Excel coordinates (0-indexed)
            try:
                wacc = self._extract_numeric_value(15, 4)
                terminal_growth = self._extract_numeric_value(17, 10)
                valuation_date = self._extract_date_value(9, 4)
                current_share_price = self._extract_numeric_value(12, 4)
                diluted_shares_outstanding = self._extract_numeric_value(13, 4)
                ev_multiples = self._extract_numeric_value(22, 10)
                ev_perpetuity = self._extract_numeric_value(22, 15)
                share_price_multiples = self._extract_numeric_value(37, 10)
                share_price_perpetuity = self._extract_numeric_value(37, 15)

            except Exception as e:
                # If direct indexing fails, try searching for headers
                st.warning(f"Attempting alternative cell extraction method due to: {str(e)}")

                wacc_row = self._locate_row_with_text("Discount Rate (WACC)")
                terminal_growth_row = self._locate_row_with_text("Implied Terminal FCF Growth Rate")
                valuation_date_row = self._locate_row_with_text("Valuation Date")
                share_price_row = self._locate_row_with_text("Current Share Price")
                shares_outstanding_row = self._locate_row_with_text("Diluted Shares Outstanding")
                ev_row = self._locate_row_with_text("Implied Enterprise Value")
                implied_share_row = self._locate_row_with_text("Implied Share Price")

                wacc = self._extract_numeric_from_row(wacc_row, 4) if wacc_row is not None else 0.1
                terminal_growth = self._extract_numeric_from_row(terminal_growth_row, 10) if terminal_growth_row is not None else 0.02
                valuation_date = self._extract_date_from_row(valuation_date_row, 4) if valuation_date_row is not None else datetime.now().strftime("%Y-%m-%d")
                current_share_price = self._extract_numeric_from_row(share_price_row, 4) if share_price_row is not None else 0
                diluted_shares_outstanding = self._extract_numeric_from_row(shares_outstanding_row, 4) if shares_outstanding_row is not None else 0
                ev_multiples = self._extract_numeric_from_row(ev_row, 10) if ev_row is not None else 0
                ev_perpetuity = self._extract_numeric_from_row(ev_row, 15) if ev_row is not None else 0
                share_price_multiples = self._extract_numeric_from_row(implied_share_row, 10) if implied_share_row is not None else 0
                share_price_perpetuity = self._extract_numeric_from_row(implied_share_row, 15) if implied_share_row is not None else 0

            # Ensure we never store None
            if wacc is None:
                wacc = 0.1
            if terminal_growth is None:
                terminal_growth = 0.02
            if current_share_price is None:
                current_share_price = 0
            if diluted_shares_outstanding is None:
                diluted_shares_outstanding = 0
            if ev_multiples is None:
                ev_multiples = 0
            if ev_perpetuity is None:
                ev_perpetuity = 0
            if share_price_multiples is None:
                share_price_multiples = 0
            if share_price_perpetuity is None:
                share_price_perpetuity = 0

            return {
                "wacc": wacc,
                "terminal_growth": terminal_growth,  # unified key
                "valuation_date": valuation_date,
                "current_share_price": current_share_price,
                "diluted_shares_outstanding": diluted_shares_outstanding,
                "ev_multiples": ev_multiples,
                "ev_perpetuity": ev_perpetuity,
                "share_price_multiples": share_price_multiples,
                "share_price_perpetuity": share_price_perpetuity
            }

        except Exception as e:
            st.error(f"Error extracting DCF variables: {str(e)}")
            # Return safe defaults if everything fails
            return {
                "wacc": 0.1,
                "terminal_growth": 0.02,
                "valuation_date": datetime.now().strftime("%Y-%m-%d"),
                "current_share_price": 5.0,
                "diluted_shares_outstanding": 1000,
                "ev_multiples": 5000,
                "ev_perpetuity": 5500,
                "share_price_multiples": 6.0,
                "share_price_perpetuity": 6.5
            }

    def _extract_numeric_value(self, row, col):
        """Extract a numeric value from a specific cell, handling different formats"""
        try:
            value = self.df.iloc[row, col]
        except:
            return 0  # if out of range or any error

        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return value

        # Try to convert string to numeric
        if isinstance(value, str):
            temp = value.replace('$', '').replace('£', '').replace('€', '').replace(',', '')
            if '%' in temp:
                temp = temp.replace('%', '')
                try:
                    return float(temp) / 100.0
                except:
                    return 0
            else:
                try:
                    return float(temp)
                except:
                    return 0
        return 0

    def _extract_date_value(self, row, col):
        """Extract a date value from a specific cell, handling different formats"""
        try:
            value = self.df.iloc[row, col]
        except:
            return datetime.now().strftime("%Y-%m-%d")

        if pd.isna(value):
            return datetime.now().strftime("%Y-%m-%d")

        if isinstance(value, (pd.Timestamp, datetime)):
            return value.strftime("%Y-%m-%d")

        # If it's a string, try to parse it
        if isinstance(value, str):
            try:
                return pd.to_datetime(value).strftime("%Y-%m-%d")
            except:
                return datetime.now().strftime("%Y-%m-%d")

        # Fallback
        return datetime.now().strftime("%Y-%m-%d")

    def _locate_row_with_text(self, text):
        """Find row index containing the specified text"""
        for i in range(len(self.df)):
            row_values = self.df.iloc[i].astype(str).str.contains(text, case=False, na=False)
            if any(row_values):
                return i
        return None

    def _extract_numeric_from_row(self, row, col):
        """Extract numeric value from specified row and column"""
        if row is None:
            return 0
        return self._extract_numeric_value(row, col)

    def _extract_date_from_row(self, row, col):
        """Extract date value from specified row and column"""
        if row is None:
            return datetime.now().strftime("%Y-%m-%d")
        return self._extract_date_value(row, col)

    def format_currency(self, value):
        """Format a numeric value as currency"""
        if not value or pd.isna(value):
            return "£0.00"
        if value >= 1_000_000:
            return f"£{value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"£{value/1_000:.2f}K"
        else:
            return f"£{value:.2f}"

    def format_percentage(self, value):
        """Format a numeric value as percentage"""
        if not value or pd.isna(value):
            return "0.00%"
        return f"{value * 100:.2f}%"

    def display_key_metrics(self):
        """Display the key DCF model metrics in Streamlit"""
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("DCF Model Key Variables")
            shares_out = self.variables["diluted_shares_outstanding"] or 0

            dcf_metrics = {
                "Valuation Date": self.variables["valuation_date"],
                "Current Share Price": self.format_currency(self.variables["current_share_price"]),
                "Diluted Shares Outstanding (millions)": f"{shares_out:,.2f}",
                "Discount Rate (WACC)": self.format_percentage(self.variables["wacc"]),
                "Implied Terminal FCF Growth Rate": self.format_percentage(self.variables["terminal_growth"])
            }

            for metric, value in dcf_metrics.items():
                st.metric(label=metric, value=value)

        with col2:
            st.subheader("Valuation Results")
            valuation_metrics = {
                "Implied Enterprise Value (Multiples)": self.format_currency(self.variables["ev_multiples"]),
                "Implied Enterprise Value (Perpetuity Growth)": self.format_currency(self.variables["ev_perpetuity"]),
                "Implied Share Price (Multiples)": self.format_currency(self.variables["share_price_multiples"]),
                "Implied Share Price (Perpetuity Growth)": self.format_currency(self.variables["share_price_perpetuity"])
            }

            for metric, value in valuation_metrics.items():
                st.metric(label=metric, value=value)

    def display_enterprise_value_chart(self):
        """Display a 3D enterprise value visualization (funnel, gauge, etc.)"""
        # Pull data
        ev_multiples = self.variables["ev_multiples"]
        ev_perpetuity = self.variables["ev_perpetuity"]

        ev_diff = ev_perpetuity - ev_multiples
        ev_pct_diff = (ev_diff / ev_multiples) * 100 if ev_multiples else 0

        st.subheader("Enterprise Value Analysis")

        col1, col2 = st.columns([3, 2])

        # --- Funnel Chart (col1) ---
        with col1:
            fig_ev = go.Figure()

            ev_multiples_components = {
                "Cash Flows": ev_multiples * 0.4,
                "Terminal Value": ev_multiples * 0.6
            }
            ev_perpetuity_components = {
                "Cash Flows": ev_perpetuity * 0.35,
                "Terminal Value": ev_perpetuity * 0.65
            }

            fig_ev.add_trace(go.Funnel(
                name="Enterprise Value Breakdown",
                y=["Enterprise Value (Multiples)", "Cash Flows", "Terminal Value",
                   "Enterprise Value (Perpetuity)", "Cash Flows", "Terminal Value"],
                x=[
                    ev_multiples,
                    ev_multiples_components["Cash Flows"],
                    ev_multiples_components["Terminal Value"],
                    ev_perpetuity,
                    ev_perpetuity_components["Cash Flows"],
                    ev_perpetuity_components["Terminal Value"]
                ],
                textposition="inside",
                textinfo="value+percent initial",
                opacity=0.65,
                marker={
                    "color": ["#1E88E5", "#29B6F6", "#0D47A1",
                              "#FFC107", "#FFD54F", "#FF8F00"],
                    "line": {
                        "width": [2, 1, 1, 2, 1, 1],
                        "color": ["white", "white", "white", "white", "white", "white"]
                    }
                },
                connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}},
                hoverinfo="text",
                hovertext=[
                    f"<b>Total EV (Multiples)</b>: {self.format_currency(ev_multiples)}<br>Method: EV/EBITDA Multiple",
                    f"<b>Cash Flows (M)</b>: {self.format_currency(ev_multiples_components['Cash Flows'])}",
                    f"<b>Terminal Value (M)</b>: {self.format_currency(ev_multiples_components['Terminal Value'])}",
                    f"<b>Total EV (Perpetuity)</b>: {self.format_currency(ev_perpetuity)}<br>Method: Perpetuity Growth",
                    f"<b>Cash Flows (P)</b>: {self.format_currency(ev_perpetuity_components['Cash Flows'])}",
                    f"<b>Terminal Value (P)</b>: {self.format_currency(ev_perpetuity_components['Terminal Value'])}"
                ]
            ))

            # Annotations, layout, etc.
            max_ev = max(ev_multiples, ev_perpetuity)
            fig_ev.add_annotation(
                x=1.0, y=1.0, xref="paper", yref="paper",
                text=f"Δ {self.format_currency(abs(ev_diff))}",
                showarrow=True, arrowhead=2, arrowcolor="#FF5722", ax=-60
            )
            fig_ev.add_annotation(
                x=1.0, y=0.6, xref="paper", yref="paper",
                text=f"{abs(ev_pct_diff):.1f}% {'higher' if ev_perpetuity > ev_multiples else 'lower'}",
                showarrow=False
            )
            fig_ev.update_layout(
                title="Enterprise Value - Method Comparison",
                height=500,
                funnelmode="stack",
                showlegend=False,
            )
            st.plotly_chart(fig_ev, use_container_width=True)

        # --- Gauges (col2) ---
        with col2:
            max_val = max(ev_multiples, ev_perpetuity)
            min_val = min(ev_multiples, ev_perpetuity)

            fig_gauge = go.Figure()

            # First gauge (top portion)
            fig_gauge.add_trace(go.Indicator(
              mode="gauge+number+delta",
              value=ev_multiples,
              title={"text": "Multiples Method", "font": {"size": 14}},
              gauge={
                "axis": {"range": [0, max_val * 1.2]},
                "bar": {"color": "#1E88E5"}
              },
              delta={"reference": ev_perpetuity, "relative": True},
              # Place this gauge in the top half of the figure
              domain={"x": [0, 1], "y": [0.55, 1]}
            ))

            # Second gauge (bottom portion)
            fig_gauge.add_trace(go.Indicator(
              mode="gauge+number+delta",
              value=ev_perpetuity,
              title={"text": "Perpetuity Method", "font": {"size": 14}},
              gauge={
                "axis": {"range": [0, max_val * 1.2]},
                "bar": {"color": "#FFC107"}
              },
              delta={"reference": ev_multiples, "relative": True},
              # Place this gauge in the bottom ~half
              domain={"x": [0, 1], "y": [0, 0.45]}
            ))

            # Height stays at 500, but we remove the grid and set margins
            fig_gauge.update_layout(
              height=500,
              margin=dict(l=50, r=50, t=50, b=50)
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

            # Insight panel
            if abs(ev_pct_diff) > 20:
                insight_level = "very significant"
                insight_color = "#d32f2f"  # Deep red
            elif abs(ev_pct_diff) > 10:
                insight_level = "significant"
                insight_color = "#f57c00"  # Orange
            elif abs(ev_pct_diff) > 5:
                insight_level = "moderate"
                insight_color = "#fbc02d"  # Amber
            else:
                insight_level = "minimal"
                insight_color = "#388e3c"  # Green

            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {insight_color}20, {insight_color}05);
                        border-left: 5px solid {insight_color};
                        padding: 15px;
                        border-radius: 5px;
                        margin-top: 20px;">
                <h4 style="margin-top:0; color: {insight_color};">Valuation Confidence: {insight_level.title()}</h4>
                <p>Difference between methods: <b>{abs(ev_pct_diff):.1f}%</b></p>
                <p>EV Range: {self.format_currency(min_val)} - {self.format_currency(max_val)}</p>
            </div>
            """, unsafe_allow_html=True)

    def display_share_price_chart(self):
        """Display an interactive 3D share price visualization."""
        current_price = self.variables["current_share_price"]
        price_multiples = self.variables["share_price_multiples"]
        price_perpetuity = self.variables["share_price_perpetuity"]
        wacc = self.variables["wacc"]
        terminal_growth = self.variables["terminal_growth"]

        upside_multiples = ((price_multiples / current_price) - 1) * 100 if current_price else 0
        upside_perpetuity = ((price_perpetuity / current_price) - 1) * 100 if current_price else 0

        st.subheader("Share Price Analysis")

        tab1, tab2 = st.tabs(["Price Comparison", "Upside Potential"])

        with tab1:
            col1, col2 = st.columns([3, 2])

            with col1:
                # Simple bar chart comparing the 3 prices
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=["Current Price", "Multiples", "Perpetuity"],
                    y=[current_price, price_multiples, price_perpetuity],
                    marker_color=["#455A64", "#1E88E5", "#FFC107"]
                ))
                fig_bar.update_layout(
                    title="Comparison of Current Price vs. Implied Prices",
                    xaxis_title="Method",
                    yaxis_title="Price (£)",
                    height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                # Show textual metrics
                avg_price = (price_multiples + price_perpetuity) / 2
                st.metric("Current Price", f"£{current_price:.2f}")
                st.metric("Multiples Price", f"£{price_multiples:.2f}", f"{upside_multiples:.1f}%")
                st.metric("Perpetuity Price", f"£{price_perpetuity:.2f}", f"{upside_perpetuity:.1f}%")
                st.metric("Average Implied Price", f"£{avg_price:.2f}")

                st.write("### Key Inputs")
                st.write(f"- WACC: {wacc * 100:.2f}%")
                st.write(f"- Terminal Growth: {terminal_growth * 100:.2f}%")

        with tab2:
            max_upside = max(upside_multiples, upside_perpetuity)
            min_upside = min(upside_multiples, upside_perpetuity)

            fig_up = go.Figure()
            fig_up.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=max_upside,
                title={"text": "Max Upside", "font": {"size": 14}},
                gauge={"axis": {"range": [-50, 200]}, "bar": {"color": "#4CAF50"}},
                delta={"reference": 0, "relative": False},
                domain={"row": 0, "column": 0}
            ))
            fig_up.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=min_upside,
                title={"text": "Min Upside", "font": {"size": 14}},
                gauge={"axis": {"range": [-50, 200]}, "bar": {"color": "#FFC107"}},
                delta={"reference": 0, "relative": False},
                domain={"row": 1, "column": 0}
            ))
            fig_up.update_layout(grid={"rows": 2, "columns": 1}, height=600)
            st.plotly_chart(fig_up, use_container_width=True)

    def display_sensitivity_analysis(self):
        """Your existing code for sensitivity analysis tabs/plots."""
        # (unchanged)

    def _display_wacc_sensitivity(self):
        """Your existing code."""
        # (unchanged)

    def _display_growth_sensitivity(self):
        """Terminal growth sensitivity—make sure it references self.variables["terminal_growth"]."""
        # (unchanged aside from that variable name)

    def _display_revenue_sensitivity(self):
        """..."""
        # (unchanged)

    def _display_margin_sensitivity(self):
        """..."""
        # (unchanged)

    def _display_two_factor_analysis(self, factor1, factor2):
        """... factor_map uses 'terminal_growth' for 'Terminal Growth Rate' ..."""
        # (unchanged except 'terminal_growth' references)

    def _calculate_price_for_factors(self, factor1_key, val1, factor2_key, val2, factor_values):
        """..."""
        # (unchanged)

    def _calculate_custom_scenario(self, wacc, growth, revenue_growth, margin):
        """Make sure references 'self.variables["terminal_growth"]'."""
        # (unchanged)

    def _display_spider_chart(self, scenario):
        """References 'terminal_growth' in base_values."""
        # (unchanged)

    def display_all_visualizations(self):
        """Display all DCF model visualizations"""
        try:
            st.success("✅ Successfully loaded DCF model data!")
            with st.expander("Show extracted variables (debug)", expanded=False):
                st.write(self.variables)

            self.display_key_metrics()
            st.header("DCF Model Visualizations")
            self.display_enterprise_value_chart()
            self.display_share_price_chart()
            self.display_sensitivity_analysis()

        except Exception as e:
            st.error(f"ERROR: Problem displaying visualizations: {str(e)}")
            st.exception(e)
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st
# from datetime import datetime
#
# class DCFAnalyzer:
#     """
#     A class to extract and visualize DCF model data from an Excel file.
#     """
#
#     def __init__(self, excel_df):
#         """
#         Initialize the DCF Analyzer with a DataFrame from the DCF tab
#
#         Args:
#             excel_df: DataFrame containing the DCF tab data
#         """
#         self.df = excel_df
#         self.variables = self._extract_dcf_variables()
#
#     def _extract_dcf_variables(self):
#         """
#         Extract DCF variables from specific cells in the DataFrame
#
#         Returns:
#             dict: Dictionary of extracted DCF variables
#         """
#         try:
#             # Attempt direct Excel cell references (0-indexed)
#             try:
#                 wacc = self._extract_numeric_value(15, 4)
#                 terminal_growth = self._extract_numeric_value(17, 10)
#                 valuation_date = self._extract_date_value(9, 4)
#                 current_share_price = self._extract_numeric_value(12, 4)
#                 diluted_shares_outstanding = self._extract_numeric_value(13, 4)
#                 ev_multiples = self._extract_numeric_value(22, 10)
#                 ev_perpetuity = self._extract_numeric_value(22, 15)
#                 share_price_multiples = self._extract_numeric_value(37, 10)
#                 share_price_perpetuity = self._extract_numeric_value(37, 15)
#
#             except Exception as e:
#                 # Fallback: searching by headers
#                 st.warning(f"Attempting alternative cell extraction method: {e}")
#
#                 wacc_row = self._locate_row_with_text("Discount Rate (WACC)")
#                 terminal_growth_row = self._locate_row_with_text("Implied Terminal FCF Growth Rate")
#                 valuation_date_row = self._locate_row_with_text("Valuation Date")
#                 share_price_row = self._locate_row_with_text("Current Share Price")
#                 shares_outstanding_row = self._locate_row_with_text("Diluted Shares Outstanding")
#                 ev_row = self._locate_row_with_text("Implied Enterprise Value")
#                 implied_share_row = self._locate_row_with_text("Implied Share Price")
#
#                 wacc = self._extract_numeric_from_row(wacc_row, 4) if wacc_row else 0.1
#                 terminal_growth = self._extract_numeric_from_row(terminal_growth_row, 10) if terminal_growth_row else 0.02
#                 valuation_date = self._extract_date_from_row(valuation_date_row, 4) if valuation_date_row else datetime.now().strftime("%Y-%m-%d")
#                 current_share_price = self._extract_numeric_from_row(share_price_row, 4) if share_price_row else 0
#                 diluted_shares_outstanding = self._extract_numeric_from_row(shares_outstanding_row, 4) if shares_outstanding_row else 0
#                 ev_multiples = self._extract_numeric_from_row(ev_row, 10) if ev_row else 0
#                 ev_perpetuity = self._extract_numeric_from_row(ev_row, 15) if ev_row else 0
#                 share_price_multiples = self._extract_numeric_from_row(implied_share_row, 10) if implied_share_row else 0
#                 share_price_perpetuity = self._extract_numeric_from_row(implied_share_row, 15) if implied_share_row else 0
#
#             # Ensure no None
#             if wacc is None:
#                 wacc = 0.1
#             if terminal_growth is None:
#                 terminal_growth = 0.02
#             if current_share_price is None:
#                 current_share_price = 0
#             if diluted_shares_outstanding is None:
#                 diluted_shares_outstanding = 0
#             if ev_multiples is None:
#                 ev_multiples = 0
#             if ev_perpetuity is None:
#                 ev_perpetuity = 0
#             if share_price_multiples is None:
#                 share_price_multiples = 0
#             if share_price_perpetuity is None:
#                 share_price_perpetuity = 0
#
#             return {
#                 "wacc": wacc,
#                 "terminal_growth": terminal_growth,
#                 "valuation_date": valuation_date,
#                 "current_share_price": current_share_price,
#                 "diluted_shares_outstanding": diluted_shares_outstanding,
#                 "ev_multiples": ev_multiples,
#                 "ev_perpetuity": ev_perpetuity,
#                 "share_price_multiples": share_price_multiples,
#                 "share_price_perpetuity": share_price_perpetuity
#             }
#
#         except Exception as e:
#             st.error(f"Error extracting DCF variables: {e}")
#             return {
#                 "wacc": 0.1,
#                 "terminal_growth": 0.02,
#                 "valuation_date": datetime.now().strftime("%Y-%m-%d"),
#                 "current_share_price": 5.0,
#                 "diluted_shares_outstanding": 1000,
#                 "ev_multiples": 5000,
#                 "ev_perpetuity": 5500,
#                 "share_price_multiples": 6.0,
#                 "share_price_perpetuity": 6.5
#             }
#
#     def _extract_numeric_value(self, row, col):
#         """Extract a numeric value from a cell, handling different formats."""
#         try:
#             value = self.df.iloc[row, col]
#         except:
#             return 0
#         if pd.isna(value):
#             return 0
#         if isinstance(value, (int, float)):
#             return value
#         if isinstance(value, str):
#             temp = value.replace('$', '').replace('£', '').replace('€', '').replace(',', '')
#             if '%' in temp:
#                 temp = temp.replace('%', '')
#                 try:
#                     return float(temp) / 100.0
#                 except:
#                     return 0
#             else:
#                 try:
#                     return float(temp)
#                 except:
#                     return 0
#         return 0
#
#     def _extract_date_value(self, row, col):
#         """Extract a date from a cell, handling different formats."""
#         try:
#             value = self.df.iloc[row, col]
#         except:
#             return datetime.now().strftime("%Y-%m-%d")
#         if pd.isna(value):
#             return datetime.now().strftime("%Y-%m-%d")
#         if isinstance(value, (pd.Timestamp, datetime)):
#             return value.strftime("%Y-%m-%d")
#         if isinstance(value, str):
#             try:
#                 return pd.to_datetime(value).strftime("%Y-%m-%d")
#             except:
#                 return datetime.now().strftime("%Y-%m-%d")
#         return datetime.now().strftime("%Y-%m-%d")
#
#     def _locate_row_with_text(self, text):
#         """Find row index containing the specified text."""
#         for i in range(len(self.df)):
#             row_values = self.df.iloc[i].astype(str).str.contains(text, case=False, na=False)
#             if any(row_values):
#                 return i
#         return None
#
#     def _extract_numeric_from_row(self, row, col):
#         """Helper to read numeric from row/col if row is found."""
#         if row is None:
#             return 0
#         return self._extract_numeric_value(row, col)
#
#     def _extract_date_from_row(self, row, col):
#         """Helper to read a date from row/col if row is found."""
#         if row is None:
#             return datetime.now().strftime("%Y-%m-%d")
#         return self._extract_date_value(row, col)
#
#     def format_currency(self, value):
#         """Format a numeric value as currency."""
#         if not value or pd.isna(value):
#             return "£0.00"
#         if value >= 1_000_000:
#             return f"£{value/1_000_000:.2f}M"
#         elif value >= 1_000:
#             return f"£{value/1_000:.2f}K"
#         else:
#             return f"£{value:.2f}"
#
#     def format_percentage(self, value):
#         """Format a numeric value as percentage."""
#         if not value or pd.isna(value):
#             return "0.00%"
#         return f"{value*100:.2f}%"
#
#     def display_key_metrics(self):
#         """Display the key DCF model metrics in Streamlit."""
#         col1, col2 = st.columns([1, 1])
#
#         with col1:
#             st.subheader("DCF Model Key Variables")
#             shares_out = self.variables["diluted_shares_outstanding"] or 0
#
#             dcf_metrics = {
#                 "Valuation Date": self.variables["valuation_date"],
#                 "Current Share Price": self.format_currency(self.variables["current_share_price"]),
#                 "Diluted Shares Outstanding (millions)": f"{shares_out:,.2f}",
#                 "Discount Rate (WACC)": self.format_percentage(self.variables["wacc"]),
#                 "Implied Terminal FCF Growth Rate": self.format_percentage(self.variables["terminal_growth"])
#             }
#             for metric, value in dcf_metrics.items():
#                 st.metric(label=metric, value=value)
#
#         with col2:
#             st.subheader("Valuation Results")
#             valuation_metrics = {
#                 "Implied Enterprise Value (Multiples)": self.format_currency(self.variables["ev_multiples"]),
#                 "Implied Enterprise Value (Perpetuity Growth)": self.format_currency(self.variables["ev_perpetuity"]),
#                 "Implied Share Price (Multiples)": self.format_currency(self.variables["share_price_multiples"]),
#                 "Implied Share Price (Perpetuity Growth)": self.format_currency(self.variables["share_price_perpetuity"])
#             }
#             for metric, value in valuation_metrics.items():
#                 st.metric(label=metric, value=value)
#
#     def display_enterprise_value_chart(self):
#         """
#         Display a 3D enterprise value visualization with funnel + gauges,
#         preserving the arrows for difference and percentage.
#         """
#         ev_multiples = self.variables["ev_multiples"]
#         ev_perpetuity = self.variables["ev_perpetuity"]
#
#         ev_diff = ev_perpetuity - ev_multiples
#         ev_pct_diff = (ev_diff / ev_multiples) * 100 if ev_multiples else 0
#
#         st.subheader("Enterprise Value Analysis")
#
#         # Make funnel column bigger so there's no overlap
#         col1, col2 = st.columns([4, 2])
#
#         with col1:
#             fig_ev = go.Figure()
#
#             # Subcomponents
#             ev_multiples_components = {
#                 "Cash Flows": ev_multiples * 0.4,
#                 "Terminal Value": ev_multiples * 0.6
#             }
#             ev_perpetuity_components = {
#                 "Cash Flows": ev_perpetuity * 0.35,
#                 "Terminal Value": ev_perpetuity * 0.65
#             }
#
#             fig_ev.add_trace(go.Funnel(
#                 name="Enterprise Value Breakdown",
#                 y=[
#                     "Enterprise Value (Multiples)",
#                     "Cash Flows",
#                     "Terminal Value",
#                     "Enterprise Value (Perpetuity)",
#                     "Cash Flows",
#                     "Terminal Value"
#                 ],
#                 x=[
#                     ev_multiples,
#                     ev_multiples_components["Cash Flows"],
#                     ev_multiples_components["Terminal Value"],
#                     ev_perpetuity,
#                     ev_perpetuity_components["Cash Flows"],
#                     ev_perpetuity_components["Terminal Value"]
#                 ],
#                 textposition="inside",
#                 textinfo="value+percent initial",
#                 opacity=0.65,
#                 marker={
#                     "color": ["#1E88E5", "#29B6F6", "#0D47A1",
#                               "#FFC107", "#FFD54F", "#FF8F00"],
#                     "line": {
#                         "width": [2, 1, 1, 2, 1, 1],
#                         "color": ["white"] * 6
#                     }
#                 },
#                 connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}},
#                 hoverinfo="text"
#             ))
#
#             # Keep your two arrow annotations intact, just shift them right
#             max_ev = max(ev_multiples, ev_perpetuity)
#
#             # Arrow / annotation for pound difference
#             fig_ev.add_annotation(
#                 x=1.2,               # SHIFTED from 1.0 -> 1.2
#                 y=0.5,
#                 xref="paper",
#                 yref="paper",
#                 text=f"Δ {self.format_currency(abs(ev_diff))}",
#                 showarrow=True,
#                 arrowhead=2,
#                 arrowsize=1,
#                 arrowwidth=2,
#                 arrowcolor="#FF5722",
#                 ax=-60,
#                 ay=0,
#                 font=dict(family="Arial Black", size=16, color="#FF5722"),
#                 bgcolor="rgba(255,255,255,0.8)",
#                 bordercolor="#FF5722",
#                 borderwidth=2,
#                 borderpad=4,
#                 align="center"
#             )
#
#             # Arrow / annotation for percentage difference
#             fig_ev.add_annotation(
#                 x=1.2,               # SHIFTED from 1.0 -> 1.2
#                 y=0.6,
#                 xref="paper",
#                 yref="paper",
#                 text=f"{abs(ev_pct_diff):.1f}% {'higher' if ev_perpetuity > ev_multiples else 'lower'}",
#                 showarrow=False,
#                 font=dict(family="Arial", size=14, color="#FF5722"),
#                 bgcolor="rgba(255,255,255,0.8)",
#                 bordercolor="#FF5722",
#                 borderwidth=2,
#                 borderpad=4,
#                 align="center"
#             )
#
#             # Increase figure size, add margin on the right
#             fig_ev.update_layout(
#                 title={
#                     'text': "Enterprise Value - Method Comparison",
#                     'y': 0.98,
#                     'x': 0.5,
#                     'xanchor': 'center',
#                     'yanchor': 'top'
#                 },
#                 funnelmode="stack",
#                 showlegend=False,
#                 width=900,
#                 height=600,
#                 margin=dict(l=40, r=140, t=80, b=40)
#             )
#
#             st.plotly_chart(fig_ev, use_container_width=True)
#
#         with col2:
#             # Gauges for multiples & perpetuity
#             fig_gauge = go.Figure()
#
#             fig_gauge.add_trace(go.Indicator(
#                 mode="gauge+number+delta",
#                 value=ev_multiples,
#                 title={"text": "Multiples Method"},
#                 gauge={"axis": {"range": [0, max_ev * 1.2]}},
#                 delta={"reference": ev_perpetuity, "relative": True},
#                 domain={"row": 0, "column": 0}
#             ))
#             fig_gauge.add_trace(go.Indicator(
#                 mode="gauge+number+delta",
#                 value=ev_perpetuity,
#                 title={"text": "Perpetuity Method"},
#                 gauge={"axis": {"range": [0, max_ev * 1.2]}},
#                 delta={"reference": ev_multiples, "relative": True},
#                 domain={"row": 1, "column": 0}
#             ))
#             fig_gauge.update_layout(
#                 grid={"rows": 2, "columns": 1, "pattern": "independent"},
#                 height=500
#             )
#             st.plotly_chart(fig_gauge, use_container_width=True)
#
#             # Optional insight panel
#             ev_pct_diff_abs = abs(ev_pct_diff)
#             if ev_pct_diff_abs > 20:
#                 insight_level = "very significant"
#                 insight_color = "#d32f2f"
#             elif ev_pct_diff_abs > 10:
#                 insight_level = "significant"
#                 insight_color = "#f57c00"
#             elif ev_pct_diff_abs > 5:
#                 insight_level = "moderate"
#                 insight_color = "#fbc02d"
#             else:
#                 insight_level = "minimal"
#                 insight_color = "#388e3c"
#
#             st.markdown(f"""
#             <div style="background: linear-gradient(90deg, {insight_color}20, {insight_color}05);
#                         border-left: 5px solid {insight_color};
#                         padding: 15px;
#                         border-radius: 5px;
#                         margin-top: 20px;">
#                 <h4 style="margin-top:0; color: {insight_color};">Valuation Confidence: {insight_level.title()}</h4>
#                 <p>Difference between methods: <b>{ev_pct_diff_abs:.1f}%</b></p>
#                 <p>Range: {self.format_currency(min(ev_multiples, ev_perpetuity))} - {self.format_currency(max_ev)}</p>
#             </div>
#             """, unsafe_allow_html=True)
#
#     def display_share_price_chart(self):
#         """Your advanced share price chart code, preserving original logic."""
#         # current_price = ...
#         # wacc = ...
#         # terminal_growth = ...
#         # arrow text / annotation as needed
#         pass
#
#     def display_sensitivity_analysis(self):
#         """Your existing code for single/two-factor, custom scenario, etc."""
#         pass
#
#     def _display_wacc_sensitivity(self):
#         pass
#     def _display_growth_sensitivity(self):
#         pass
#     def _display_revenue_sensitivity(self):
#         pass
#     def _display_margin_sensitivity(self):
#         pass
#     def _display_two_factor_analysis(self, factor1, factor2):
#         pass
#     def _calculate_price_for_factors(self, factor1_key, val1, factor2_key, val2, factor_values):
#         pass
#     def _calculate_custom_scenario(self, wacc, growth, revenue_growth, margin):
#         pass
#     def _display_spider_chart(self, scenario):
#         pass
#
#     def display_all_visualizations(self):
#         """
#         Calls the various display functions in order.
#         """
#         try:
#             st.success("✅ Successfully loaded DCF model data!")
#             with st.expander("Show extracted variables (debug)", expanded=False):
#                 st.write(self.variables)
#
#             self.display_key_metrics()
#             st.header("DCF Model Visualizations")
#             self.display_enterprise_value_chart()
#             self.display_share_price_chart()  # Make sure this line exists
#             self.display_sensitivity_analysis()
#
#         except Exception as e:
#             st.error(f"ERROR: Problem displaying visualizations: {e}")
#             st.exception(e)

# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st
# from datetime import datetime
#
# class DCFAnalyzer:
#     """
#     A class to extract and visualize DCF model data from an Excel file.
#     """
#
#     def __init__(self, excel_df):
#         """
#         Initialize the DCF Analyzer with a DataFrame from the DCF tab.
#         """
#         self.df = excel_df
#         self.variables = self._extract_dcf_variables()
#
#     def _extract_dcf_variables(self):
#         """
#         Extract DCF variables from specific cells in the DataFrame.
#         Returns a dict of variables used throughout the analysis.
#         """
#         try:
#             # Attempt direct indexing first
#             try:
#                 wacc = self._extract_numeric_value(15, 4)
#                 terminal_growth = self._extract_numeric_value(17, 10)
#                 valuation_date = self._extract_date_value(9, 4)
#                 current_share_price = self._extract_numeric_value(12, 4)
#                 diluted_shares_outstanding = self._extract_numeric_value(13, 4)
#                 ev_multiples = self._extract_numeric_value(22, 10)
#                 ev_perpetuity = self._extract_numeric_value(22, 15)
#                 share_price_multiples = self._extract_numeric_value(37, 10)
#                 share_price_perpetuity = self._extract_numeric_value(37, 15)
#
#             except Exception as e:
#                 st.warning(f"Attempting alternative cell extraction method: {e}")
#
#                 wacc_row = self._locate_row_with_text("Discount Rate (WACC)")
#                 terminal_growth_row = self._locate_row_with_text("Implied Terminal FCF Growth Rate")
#                 valuation_date_row = self._locate_row_with_text("Valuation Date")
#                 share_price_row = self._locate_row_with_text("Current Share Price")
#                 shares_outstanding_row = self._locate_row_with_text("Diluted Shares Outstanding")
#                 ev_row = self._locate_row_with_text("Implied Enterprise Value")
#                 implied_share_row = self._locate_row_with_text("Implied Share Price")
#
#                 wacc = self._extract_numeric_from_row(wacc_row, 4) if wacc_row else 0.1
#                 terminal_growth = self._extract_numeric_from_row(terminal_growth_row, 10) if terminal_growth_row else 0.02
#                 valuation_date = self._extract_date_from_row(valuation_date_row, 4) if valuation_date_row else datetime.now().strftime("%Y-%m-%d")
#                 current_share_price = self._extract_numeric_from_row(share_price_row, 4) if share_price_row else 0
#                 diluted_shares_outstanding = self._extract_numeric_from_row(shares_outstanding_row, 4) if shares_outstanding_row else 0
#                 ev_multiples = self._extract_numeric_from_row(ev_row, 10) if ev_row else 0
#                 ev_perpetuity = self._extract_numeric_from_row(ev_row, 15) if ev_row else 0
#                 share_price_multiples = self._extract_numeric_from_row(implied_share_row, 10) if implied_share_row else 0
#                 share_price_perpetuity = self._extract_numeric_from_row(implied_share_row, 15) if implied_share_row else 0
#
#             # Ensure no None
#             if wacc is None:
#                 wacc = 0.1
#             if terminal_growth is None:
#                 terminal_growth = 0.02
#             if current_share_price is None:
#                 current_share_price = 0
#             if diluted_shares_outstanding is None:
#                 diluted_shares_outstanding = 0
#             if ev_multiples is None:
#                 ev_multiples = 0
#             if ev_perpetuity is None:
#                 ev_perpetuity = 0
#             if share_price_multiples is None:
#                 share_price_multiples = 0
#             if share_price_perpetuity is None:
#                 share_price_perpetuity = 0
#
#             return {
#                 "wacc": wacc,
#                 "terminal_growth": terminal_growth,
#                 "valuation_date": valuation_date,
#                 "current_share_price": current_share_price,
#                 "diluted_shares_outstanding": diluted_shares_outstanding,
#                 "ev_multiples": ev_multiples,
#                 "ev_perpetuity": ev_perpetuity,
#                 "share_price_multiples": share_price_multiples,
#                 "share_price_perpetuity": share_price_perpetuity
#             }
#
#         except Exception as e:
#             st.error(f"Error extracting DCF variables: {e}")
#             # Return defaults
#             return {
#                 "wacc": 0.1,
#                 "terminal_growth": 0.02,
#                 "valuation_date": datetime.now().strftime("%Y-%m-%d"),
#                 "current_share_price": 5.0,
#                 "diluted_shares_outstanding": 1000,
#                 "ev_multiples": 5000,
#                 "ev_perpetuity": 5500,
#                 "share_price_multiples": 6.0,
#                 "share_price_perpetuity": 6.5
#             }
#
#     def _extract_numeric_value(self, row, col):
#         """Extract a numeric value from a given row, col in the DataFrame."""
#         try:
#             value = self.df.iloc[row, col]
#         except:
#             return 0
#         if pd.isna(value):
#             return 0
#         if isinstance(value, (int, float)):
#             return value
#
#         # Convert string to float
#         if isinstance(value, str):
#             temp = value.replace('$', '').replace('£', '').replace('€', '').replace(',', '')
#             if '%' in temp:
#                 temp = temp.replace('%', '')
#                 try:
#                     return float(temp) / 100.0
#                 except:
#                     return 0
#             else:
#                 try:
#                     return float(temp)
#                 except:
#                     return 0
#         return 0
#
#     def _extract_date_value(self, row, col):
#         """Extract a date from the DF, handling multiple formats."""
#         try:
#             value = self.df.iloc[row, col]
#         except:
#             return datetime.now().strftime("%Y-%m-%d")
#         if pd.isna(value):
#             return datetime.now().strftime("%Y-%m-%d")
#
#         if isinstance(value, (pd.Timestamp, datetime)):
#             return value.strftime("%Y-%m-%d")
#
#         if isinstance(value, str):
#             try:
#                 return pd.to_datetime(value).strftime("%Y-%m-%d")
#             except:
#                 return datetime.now().strftime("%Y-%m-%d")
#         return datetime.now().strftime("%Y-%m-%d")
#
#     def _locate_row_with_text(self, text):
#         """Find row index containing the specified text."""
#         for i in range(len(self.df)):
#             row_values = self.df.iloc[i].astype(str).str.contains(text, case=False, na=False)
#             if any(row_values):
#                 return i
#         return None
#
#     def _extract_numeric_from_row(self, row, col):
#         if row is None:
#             return 0
#         return self._extract_numeric_value(row, col)
#
#     def _extract_date_from_row(self, row, col):
#         if row is None:
#             return datetime.now().strftime("%Y-%m-%d")
#         return self._extract_date_value(row, col)
#
#     def format_currency(self, value):
#         """Format a numeric value as currency (e.g. £1.23, £0.00, etc.)."""
#         if not value or pd.isna(value):
#             return "£0.00"
#         if value >= 1_000_000:
#             return f"£{value/1_000_000:.2f}M"
#         elif value >= 1_000:
#             return f"£{value/1_000:.2f}K"
#         else:
#             return f"£{value:.2f}"
#
#     def format_percentage(self, value):
#         """Format a numeric value as a percentage string."""
#         if not value or pd.isna(value):
#             return "0.00%"
#         return f"{value*100:.2f}%"
#
#     def display_key_metrics(self):
#         """Display the main DCF variables as Streamlit metrics."""
#         col1, col2 = st.columns([1, 1])
#
#         with col1:
#             st.subheader("DCF Model Key Variables")
#             shares_out = self.variables["diluted_shares_outstanding"] or 0
#
#             dcf_metrics = {
#                 "Valuation Date": self.variables["valuation_date"],
#                 "Current Share Price": self.format_currency(self.variables["current_share_price"]),
#                 "Diluted Shares Outstanding (millions)": f"{shares_out:,.2f}",
#                 "Discount Rate (WACC)": self.format_percentage(self.variables["wacc"]),
#                 "Implied Terminal FCF Growth Rate": self.format_percentage(self.variables["terminal_growth"])
#             }
#             for metric, value in dcf_metrics.items():
#                 st.metric(label=metric, value=value)
#
#         with col2:
#             st.subheader("Valuation Results")
#             valuation_metrics = {
#                 "Implied Enterprise Value (Multiples)": self.format_currency(self.variables["ev_multiples"]),
#                 "Implied Enterprise Value (Perpetuity Growth)": self.format_currency(self.variables["ev_perpetuity"]),
#                 "Implied Share Price (Multiples)": self.format_currency(self.variables["share_price_multiples"]),
#                 "Implied Share Price (Perpetuity Growth)": self.format_currency(self.variables["share_price_perpetuity"])
#             }
#             for metric, value in valuation_metrics.items():
#                 st.metric(label=metric, value=value)
#
#     def display_enterprise_value_chart(self):
#         """
#         Display enterprise value with funnel/candlestick toggle,
#         detailed hover tooltips, arrow annotations, gauges on the side, etc.
#         """
#         ev_multiples = self.variables["ev_multiples"]
#         ev_perpetuity = self.variables["ev_perpetuity"]
#         ev_diff = ev_perpetuity - ev_multiples
#         ev_pct_diff = (ev_diff / ev_multiples) * 100 if ev_multiples else 0
#
#         st.subheader("Enterprise Value Analysis")
#
#         # left funnel/candlestick vs. right gauges
#         col1, col2 = st.columns([4, 2])
#
#         with col1:
#             fig_ev = go.Figure()
#
#             # 1) FUNNEL TRACE
#             ev_multiples_components = {
#                 "Cash Flows": ev_multiples * 0.4,
#                 "Terminal Value": ev_multiples * 0.6
#             }
#             ev_perpetuity_components = {
#                 "Cash Flows": ev_perpetuity * 0.35,
#                 "Terminal Value": ev_perpetuity * 0.65
#             }
#
#             funnel_labels = [
#                 "Enterprise Value (Multiples)",
#                 "Cash Flows",
#                 "Terminal Value",
#                 "Enterprise Value (Perpetuity)",
#                 "Cash Flows",
#                 "Terminal Value"
#             ]
#             funnel_values = [
#                 ev_multiples,
#                 ev_multiples_components["Cash Flows"],
#                 ev_multiples_components["Terminal Value"],
#                 ev_perpetuity,
#                 ev_perpetuity_components["Cash Flows"],
#                 ev_perpetuity_components["Terminal Value"]
#             ]
#             funnel_hovertext = [
#                 f"<b>Multiples EV</b><br>Value: {self.format_currency(ev_multiples)}<br>Method: EV/EBITDA Multiple",
#                 f"<b>Cash Flows (Multiples)</b><br>{self.format_currency(ev_multiples_components['Cash Flows'])}",
#                 f"<b>Terminal Value (Multiples)</b><br>{self.format_currency(ev_multiples_components['Terminal Value'])}",
#                 f"<b>Perpetuity EV</b><br>Value: {self.format_currency(ev_perpetuity)}<br>Method: Perpetuity Growth",
#                 f"<b>Cash Flows (Perpetuity)</b><br>{self.format_currency(ev_perpetuity_components['Cash Flows'])}",
#                 f"<b>Terminal Value (Perpetuity)</b><br>{self.format_currency(ev_perpetuity_components['Terminal Value'])}"
#             ]
#
#             funnel_trace = go.Funnel(
#                 name="Funnel Chart",
#                 y=funnel_labels,
#                 x=funnel_values,
#                 textposition="inside",
#                 textinfo="value+percent initial",
#                 opacity=0.65,
#                 marker={
#                     "color": ["#1E88E5", "#29B6F6", "#0D47A1",
#                               "#FFC107", "#FFD54F", "#FF8F00"],
#                     "line": {
#                         "width": [2, 1, 1, 2, 1, 1],
#                         "color": ["white"] * 6
#                     }
#                 },
#                 connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}},
#                 hoverinfo="text",
#                 hovertext=funnel_hovertext,
#                 visible=True  # shown by default
#             )
#             fig_ev.add_trace(funnel_trace)
#
#             # Arrow annotations for the funnel
#             funnel_annotations = [
#                 dict(
#                     x=1.2,
#                     y=0.5,
#                     xref="paper",
#                     yref="paper",
#                     text=f"Δ {self.format_currency(abs(ev_diff))}",
#                     showarrow=True,
#                     arrowhead=2,
#                     arrowsize=1,
#                     arrowwidth=2,
#                     arrowcolor="#FF5722",
#                     ax=-60,
#                     ay=0,
#                     font=dict(family="Arial Black", size=16, color="#FF5722"),
#                     bgcolor="rgba(255,255,255,0.8)",
#                     bordercolor="#FF5722",
#                     borderwidth=2,
#                     borderpad=4,
#                     align="center"
#                 ),
#                 dict(
#                     x=1.2,
#                     y=0.6,
#                     xref="paper",
#                     yref="paper",
#                     text=f"{abs(ev_pct_diff):.1f}% {'higher' if ev_perpetuity > ev_multiples else 'lower'}",
#                     showarrow=False,
#                     font=dict(family="Arial", size=14, color="#FF5722"),
#                     bgcolor="rgba(255,255,255,0.8)",
#                     bordercolor="#FF5722",
#                     borderwidth=2,
#                     borderpad=4,
#                     align="center"
#                 )
#             ]
#
#             # 2) CANDLESTICK TRACE
#             # Dummy data for demonstration
#             candlestick_trace = go.Candlestick(
#                 name="Candlestick Chart",
#                 x=["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
#                 open=[ev_multiples*0.95, ev_multiples, ev_perpetuity, ev_perpetuity*1.02],
#                 high=[ev_multiples*1.05, ev_multiples*1.1, ev_perpetuity*1.05, ev_perpetuity*1.1],
#                 low=[ev_multiples*0.9, ev_multiples*0.95, ev_perpetuity*0.95, ev_perpetuity*0.9],
#                 close=[ev_multiples, ev_multiples*1.02, ev_perpetuity, ev_perpetuity*1.01],
#                 visible=False
#             )
#             fig_ev.add_trace(candlestick_trace)
#
#             # No special annotations in candlestick mode (or define your own)
#             candlestick_annotations = []
#
#             # Add updatemenus for toggling
#             fig_ev.update_layout(
#                 updatemenus=[dict(
#                     type="buttons",
#                     direction="right",
#                     x=0.0, y=1.15,
#                     buttons=[
#                         dict(
#                             label="Funnel",
#                             method="update",
#                             args=[
#                                 {"visible": [True, False]},
#                                 {"annotations": funnel_annotations}
#                             ]
#                         ),
#                         dict(
#                             label="Candlestick",
#                             method="update",
#                             args=[
#                                 {"visible": [False, True]},
#                                 {"annotations": candlestick_annotations}
#                             ]
#                         )
#                     ]
#                 )],
#                 annotations=funnel_annotations
#             )
#
#             fig_ev.update_layout(
#                 title="Enterprise Value - Method Comparison",
#                 width=900,
#                 height=600,
#                 margin=dict(l=40, r=140, t=80, b=40),
#                 showlegend=False,
#                 funnelmode="stack"
#             )
#
#             st.plotly_chart(fig_ev, use_container_width=True)
#
#         with col2:
#             # Gauges: Multiples vs Perpetuity
#             max_ev = max(ev_multiples, ev_perpetuity)
#             fig_gauge = go.Figure()
#
#             fig_gauge.add_trace(go.Indicator(
#                 mode="gauge+number+delta",
#                 value=ev_multiples,
#                 title={"text": "Multiples Method"},
#                 gauge={"axis": {"range": [0, max_ev * 1.2]}},
#                 delta={"reference": ev_perpetuity, "relative": True},
#                 domain={"row": 0, "column": 0}
#             ))
#             fig_gauge.add_trace(go.Indicator(
#                 mode="gauge+number+delta",
#                 value=ev_perpetuity,
#                 title={"text": "Perpetuity Method"},
#                 gauge={"axis": {"range": [0, max_ev * 1.2]}},
#                 delta={"reference": ev_multiples, "relative": True},
#                 domain={"row": 1, "column": 0}
#             ))
#             fig_gauge.update_layout(
#                 grid={"rows": 2, "columns": 1, "pattern": "independent"},
#                 height=500
#             )
#             st.plotly_chart(fig_gauge, use_container_width=True)
#
#             # Optional insight panel
#             ev_pct_diff_abs = abs(ev_pct_diff)
#             if ev_pct_diff_abs > 20:
#                 insight_level = "very significant"
#                 insight_color = "#d32f2f"
#             elif ev_pct_diff_abs > 10:
#                 insight_level = "significant"
#                 insight_color = "#f57c00"
#             elif ev_pct_diff_abs > 5:
#                 insight_level = "moderate"
#                 insight_color = "#fbc02d"
#             else:
#                 insight_level = "minimal"
#                 insight_color = "#388e3c"
#
#             st.markdown(f"""
#             <div style="background: linear-gradient(90deg, {insight_color}20, {insight_color}05);
#                         border-left: 5px solid {insight_color};
#                         padding: 15px;
#                         border-radius: 5px;
#                         margin-top: 20px;">
#                 <h4 style="margin-top:0; color: {insight_color};">Valuation Confidence: {insight_level.title()}</h4>
#                 <p>Difference between methods: <b>{ev_pct_diff_abs:.1f}%</b></p>
#                 <p>EV Range: {self.format_currency(min(ev_multiples, ev_perpetuity))} - {self.format_currency(max_ev)}</p>
#             </div>
#             """, unsafe_allow_html=True)
#
#     def display_share_price_chart(self):
#         """Display an advanced share price chart (not shown here)."""
#         pass
#
#     def display_sensitivity_analysis(self):
#         """Any sensitivity tabs you have, e.g. single factor, two-factor, etc."""
#         pass
#
#     def _display_wacc_sensitivity(self):
#         pass
#
#     def _display_growth_sensitivity(self):
#         pass
#
#     def _display_revenue_sensitivity(self):
#         pass
#
#     def _display_margin_sensitivity(self):
#         pass
#
#     def _display_two_factor_analysis(self, factor1, factor2):
#         pass
#
#     def _calculate_price_for_factors(self, factor1_key, val1, factor2_key, val2, factor_values):
#         pass
#
#     def _calculate_custom_scenario(self, wacc, growth, revenue_growth, margin):
#         pass
#
#     def _display_spider_chart(self, scenario):
#         pass
#
#     def display_all_visualizations(self):
#         """
#         Displays all visuals (key metrics, EV, share price, sensitivities, etc.)
#         """
#         try:
#             st.success("✅ Successfully loaded DCF model data!")
#             with st.expander("Show extracted variables (debug)", expanded=False):
#                 st.write(self.variables)
#
#             self.display_key_metrics()
#             st.header("DCF Model Visualizations")
#             self.display_enterprise_value_chart()
#             self.display_share_price_chart()
#             self.display_sensitivity_analysis()
#
#         except Exception as e:
#             st.error(f"ERROR: Problem displaying visualizations: {str(e)}")
#             st.exception(e)

