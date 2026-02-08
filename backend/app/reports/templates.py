"""
Report Templates for RUDRA Platform

Defines standard report structures for regulatory compliance:
- Basel III compliance reports
- CCAR-style stress test reports
- Systemic risk assessment reports
"""

from typing import Any, Dict, List


# Basel III Compliance Report Template
BASEL_III_TEMPLATE = {
    "title": "Basel III Compliance Report",
    "sections": [
        {
            "id": "executive_summary",
            "name": "Executive Summary",
            "description": "High-level overview of Basel III compliance status",
            "required_fields": [
                "report_date",
                "institution_name",
                "reporting_period",
                "overall_compliance_status"
            ],
            "subsections": [
                {
                    "id": "key_findings",
                    "name": "Key Findings",
                    "fields": ["findings_summary"]
                },
                {
                    "id": "recommendations",
                    "name": "Recommendations",
                    "fields": ["recommendations_list"]
                }
            ]
        },
        {
            "id": "capital_adequacy",
            "name": "Capital Adequacy",
            "description": "Analysis of capital ratios and requirements",
            "required_fields": [
                "common_equity_tier1_ratio",
                "tier1_capital_ratio",
                "total_capital_ratio",
                "leverage_ratio"
            ],
            "metrics": {
                "cet1_ratio": {
                    "name": "Common Equity Tier 1 (CET1) Ratio",
                    "minimum_requirement": 4.5,
                    "target_requirement": 7.0,
                    "unit": "percent",
                    "description": "CET1 capital as a percentage of risk-weighted assets"
                },
                "tier1_ratio": {
                    "name": "Tier 1 Capital Ratio",
                    "minimum_requirement": 6.0,
                    "target_requirement": 8.5,
                    "unit": "percent",
                    "description": "Tier 1 capital as a percentage of risk-weighted assets"
                },
                "total_capital_ratio": {
                    "name": "Total Capital Ratio",
                    "minimum_requirement": 8.0,
                    "target_requirement": 10.5,
                    "unit": "percent",
                    "description": "Total regulatory capital as a percentage of risk-weighted assets"
                },
                "leverage_ratio": {
                    "name": "Leverage Ratio",
                    "minimum_requirement": 3.0,
                    "target_requirement": 5.0,
                    "unit": "percent",
                    "description": "Tier 1 capital as a percentage of total exposure"
                }
            },
            "subsections": [
                {
                    "id": "capital_components",
                    "name": "Capital Components",
                    "fields": [
                        "cet1_capital",
                        "additional_tier1_capital",
                        "tier2_capital",
                        "total_capital"
                    ]
                },
                {
                    "id": "risk_weighted_assets",
                    "name": "Risk-Weighted Assets",
                    "fields": [
                        "credit_risk_rwa",
                        "market_risk_rwa",
                        "operational_risk_rwa",
                        "total_rwa"
                    ]
                },
                {
                    "id": "capital_buffers",
                    "name": "Capital Buffers",
                    "fields": [
                        "capital_conservation_buffer",
                        "countercyclical_buffer",
                        "g_sib_buffer",
                        "total_buffer_requirement"
                    ]
                }
            ]
        },
        {
            "id": "liquidity_coverage",
            "name": "Liquidity Coverage Ratio (LCR)",
            "description": "Assessment of short-term liquidity resilience",
            "required_fields": [
                "lcr_ratio",
                "high_quality_liquid_assets",
                "net_cash_outflows"
            ],
            "metrics": {
                "lcr_ratio": {
                    "name": "Liquidity Coverage Ratio",
                    "minimum_requirement": 100.0,
                    "target_requirement": 110.0,
                    "unit": "percent",
                    "description": "HQLA as a percentage of net cash outflows over 30 days"
                }
            },
            "subsections": [
                {
                    "id": "hqla_composition",
                    "name": "High-Quality Liquid Assets",
                    "fields": [
                        "level1_assets",
                        "level2a_assets",
                        "level2b_assets",
                        "total_hqla"
                    ]
                },
                {
                    "id": "cash_flows",
                    "name": "Cash Flow Analysis",
                    "fields": [
                        "cash_outflows",
                        "cash_inflows",
                        "net_cash_outflows"
                    ]
                }
            ]
        },
        {
            "id": "net_stable_funding",
            "name": "Net Stable Funding Ratio (NSFR)",
            "description": "Assessment of long-term funding stability",
            "required_fields": [
                "nsfr_ratio",
                "available_stable_funding",
                "required_stable_funding"
            ],
            "metrics": {
                "nsfr_ratio": {
                    "name": "Net Stable Funding Ratio",
                    "minimum_requirement": 100.0,
                    "target_requirement": 105.0,
                    "unit": "percent",
                    "description": "Available stable funding as a percentage of required stable funding"
                }
            }
        },
        {
            "id": "large_exposures",
            "name": "Large Exposures",
            "description": "Analysis of concentration risk",
            "required_fields": [
                "number_of_large_exposures",
                "largest_exposure_limit_utilization"
            ],
            "metrics": {
                "single_exposure_limit": {
                    "name": "Single Exposure Limit",
                    "maximum_limit": 25.0,
                    "unit": "percent_of_capital",
                    "description": "Maximum exposure to a single counterparty"
                }
            }
        },
        {
            "id": "risk_management",
            "name": "Risk Management Assessment",
            "description": "Evaluation of risk management framework",
            "required_fields": [
                "credit_risk_assessment",
                "market_risk_assessment",
                "operational_risk_assessment",
                "liquidity_risk_assessment"
            ]
        },
        {
            "id": "disclosure",
            "name": "Pillar 3 Disclosure Requirements",
            "description": "Market discipline and transparency requirements",
            "required_fields": [
                "disclosure_compliance",
                "last_disclosure_date",
                "next_disclosure_date"
            ]
        }
    ],
    "appendices": [
        {
            "id": "methodology",
            "name": "Methodology and Assumptions",
            "description": "Detailed methodology used in calculations"
        },
        {
            "id": "regulatory_references",
            "name": "Regulatory References",
            "description": "References to applicable regulations"
        }
    ]
}


# CCAR-Style Stress Test Report Template
STRESS_TEST_TEMPLATE = {
    "title": "Comprehensive Capital Analysis and Review (CCAR) - Stress Test Report",
    "sections": [
        {
            "id": "executive_summary",
            "name": "Executive Summary",
            "description": "Overview of stress test results",
            "required_fields": [
                "test_date",
                "scenario_description",
                "institutions_tested",
                "pass_fail_summary"
            ]
        },
        {
            "id": "scenarios",
            "name": "Stress Test Scenarios",
            "description": "Description of stress scenarios applied",
            "required_fields": [
                "baseline_scenario",
                "adverse_scenario",
                "severely_adverse_scenario"
            ],
            "subsections": [
                {
                    "id": "macroeconomic_assumptions",
                    "name": "Macroeconomic Assumptions",
                    "fields": [
                        "gdp_growth",
                        "unemployment_rate",
                        "inflation_rate",
                        "interest_rates",
                        "equity_market_decline",
                        "real_estate_decline"
                    ]
                },
                {
                    "id": "financial_market_assumptions",
                    "name": "Financial Market Assumptions",
                    "fields": [
                        "credit_spreads",
                        "volatility_index",
                        "foreign_exchange_rates",
                        "commodity_prices"
                    ]
                },
                {
                    "id": "idiosyncratic_shocks",
                    "name": "Idiosyncratic Shocks",
                    "fields": [
                        "counterparty_defaults",
                        "operational_failures",
                        "market_disruptions"
                    ]
                }
            ]
        },
        {
            "id": "capital_projections",
            "name": "Capital Projections",
            "description": "Projected capital ratios under stress scenarios",
            "required_fields": [
                "pre_stress_capital_ratios",
                "post_stress_capital_ratios",
                "minimum_capital_ratios",
                "capital_action_assumptions"
            ],
            "metrics": {
                "cet1_ratio_stressed": {
                    "name": "CET1 Ratio (Stressed)",
                    "minimum_requirement": 4.5,
                    "description": "Minimum CET1 ratio during stress period"
                },
                "tier1_ratio_stressed": {
                    "name": "Tier 1 Ratio (Stressed)",
                    "minimum_requirement": 6.0,
                    "description": "Minimum Tier 1 ratio during stress period"
                },
                "total_capital_ratio_stressed": {
                    "name": "Total Capital Ratio (Stressed)",
                    "minimum_requirement": 8.0,
                    "description": "Minimum total capital ratio during stress period"
                }
            },
            "subsections": [
                {
                    "id": "quarterly_projections",
                    "name": "Quarterly Capital Projections",
                    "fields": [
                        "q1_projections",
                        "q2_projections",
                        "q3_projections",
                        "q4_projections"
                    ]
                },
                {
                    "id": "capital_actions",
                    "name": "Planned Capital Actions",
                    "fields": [
                        "dividend_payments",
                        "share_repurchases",
                        "capital_issuances"
                    ]
                }
            ]
        },
        {
            "id": "loss_projections",
            "name": "Loss Projections",
            "description": "Projected losses under stress scenarios",
            "required_fields": [
                "total_losses",
                "pre_provision_net_revenue",
                "provision_for_loan_losses",
                "realized_losses_securities",
                "trading_market_risk_losses"
            ],
            "subsections": [
                {
                    "id": "credit_losses",
                    "name": "Credit Losses",
                    "fields": [
                        "commercial_real_estate_losses",
                        "residential_mortgage_losses",
                        "credit_card_losses",
                        "corporate_loan_losses",
                        "other_consumer_losses"
                    ]
                },
                {
                    "id": "market_losses",
                    "name": "Market Risk Losses",
                    "fields": [
                        "trading_book_losses",
                        "available_for_sale_losses",
                        "counterparty_credit_losses"
                    ]
                },
                {
                    "id": "operational_losses",
                    "name": "Operational Risk Losses",
                    "fields": [
                        "fraud_losses",
                        "litigation_losses",
                        "cyber_risk_losses",
                        "other_operational_losses"
                    ]
                }
            ]
        },
        {
            "id": "revenue_projections",
            "name": "Revenue Projections",
            "description": "Projected revenues under stress scenarios",
            "required_fields": [
                "net_interest_income",
                "non_interest_income",
                "trading_revenue",
                "fee_income"
            ]
        },
        {
            "id": "risk_weighted_assets",
            "name": "Risk-Weighted Assets Projections",
            "description": "Projected RWA under stress scenarios",
            "required_fields": [
                "credit_risk_rwa_projection",
                "market_risk_rwa_projection",
                "operational_risk_rwa_projection",
                "total_rwa_projection"
            ]
        },
        {
            "id": "liquidity_stress",
            "name": "Liquidity Stress Assessment",
            "description": "Liquidity position under stress",
            "required_fields": [
                "stressed_lcr",
                "cash_flow_projections",
                "funding_sources_stressed",
                "contingency_funding_plan"
            ]
        },
        {
            "id": "counterparty_risk",
            "name": "Counterparty Credit Risk",
            "description": "Analysis of counterparty exposures under stress",
            "required_fields": [
                "derivative_exposures",
                "securities_financing_exposures",
                "clearing_member_exposures",
                "largest_counterparty_losses"
            ]
        },
        {
            "id": "results_analysis",
            "name": "Results and Analysis",
            "description": "Detailed analysis of stress test results",
            "required_fields": [
                "overall_assessment",
                "vulnerabilities_identified",
                "mitigating_actions",
                "sensitivity_analysis"
            ]
        },
        {
            "id": "governance",
            "name": "Governance and Controls",
            "description": "Stress testing governance framework",
            "required_fields": [
                "methodology_approval",
                "model_validation",
                "independent_review",
                "board_oversight"
            ]
        }
    ],
    "appendices": [
        {
            "id": "methodology",
            "name": "Stress Testing Methodology",
            "description": "Detailed methodology and modeling approach"
        },
        {
            "id": "assumptions",
            "name": "Key Assumptions and Limitations",
            "description": "Critical assumptions and model limitations"
        },
        {
            "id": "sensitivity",
            "name": "Sensitivity Analysis",
            "description": "Alternative scenario results"
        }
    ]
}


# Systemic Risk Assessment Report Template
SYSTEMIC_RISK_TEMPLATE = {
    "title": "Systemic Risk Assessment Report",
    "sections": [
        {
            "id": "executive_summary",
            "name": "Executive Summary",
            "description": "Overview of systemic risk assessment",
            "required_fields": [
                "assessment_date",
                "risk_level",
                "key_vulnerabilities",
                "recommended_actions"
            ]
        },
        {
            "id": "network_analysis",
            "name": "Network Structure Analysis",
            "description": "Analysis of financial network topology",
            "required_fields": [
                "network_size",
                "network_density",
                "clustering_coefficient",
                "average_path_length",
                "degree_distribution"
            ],
            "metrics": {
                "network_density": {
                    "name": "Network Density",
                    "description": "Proportion of actual connections to possible connections",
                    "interpretation": "Higher density indicates more interconnectedness"
                },
                "clustering_coefficient": {
                    "name": "Clustering Coefficient",
                    "description": "Degree to which nodes cluster together",
                    "interpretation": "Higher clustering may amplify shocks locally"
                }
            },
            "subsections": [
                {
                    "id": "centrality_measures",
                    "name": "Centrality Measures",
                    "fields": [
                        "degree_centrality",
                        "betweenness_centrality",
                        "eigenvector_centrality",
                        "pagerank"
                    ]
                },
                {
                    "id": "systemically_important_nodes",
                    "name": "Systemically Important Institutions",
                    "fields": [
                        "g_sib_list",
                        "d_sib_list",
                        "interconnectedness_scores"
                    ]
                }
            ]
        },
        {
            "id": "contagion_analysis",
            "name": "Contagion and Cascade Analysis",
            "description": "Assessment of contagion risk and cascade potential",
            "required_fields": [
                "cascade_probability",
                "expected_cascade_depth",
                "cascade_amplification_factor",
                "system_vulnerability_index"
            ],
            "metrics": {
                "cascade_probability": {
                    "name": "Cascade Probability",
                    "description": "Probability of a cascade event occurring",
                    "risk_thresholds": {
                        "low": 0.1,
                        "medium": 0.3,
                        "high": 0.5
                    }
                },
                "expected_cascade_depth": {
                    "name": "Expected Cascade Depth",
                    "description": "Average number of cascade rounds",
                    "risk_thresholds": {
                        "low": 2,
                        "medium": 4,
                        "high": 6
                    }
                }
            },
            "subsections": [
                {
                    "id": "shock_scenarios",
                    "name": "Shock Propagation Scenarios",
                    "fields": [
                        "single_institution_failure",
                        "multiple_institution_failure",
                        "sector_specific_shock",
                        "market_wide_shock"
                    ]
                },
                {
                    "id": "cascade_paths",
                    "name": "Critical Cascade Paths",
                    "fields": [
                        "primary_cascade_paths",
                        "secondary_cascade_paths",
                        "systemically_critical_links"
                    ]
                }
            ]
        },
        {
            "id": "interconnectedness",
            "name": "Interconnectedness Assessment",
            "description": "Evaluation of institutional interconnections",
            "required_fields": [
                "total_exposures",
                "exposure_concentration",
                "bilateral_exposure_matrix",
                "common_exposure_analysis"
            ],
            "subsections": [
                {
                    "id": "exposure_metrics",
                    "name": "Exposure Metrics",
                    "fields": [
                        "gross_exposures",
                        "net_exposures",
                        "exposure_to_capital_ratios",
                        "largest_exposures"
                    ]
                },
                {
                    "id": "indirect_exposures",
                    "name": "Indirect Exposures",
                    "fields": [
                        "second_order_exposures",
                        "common_asset_holdings",
                        "funding_interdependencies"
                    ]
                }
            ]
        },
        {
            "id": "systemic_risk_indicators",
            "name": "Systemic Risk Indicators",
            "description": "Key systemic risk metrics and indicators",
            "required_fields": [
                "systemic_expected_shortfall",
                "marginal_expected_shortfall",
                "distress_insurance_premium",
                "absorption_ratio"
            ],
            "metrics": {
                "systemic_expected_shortfall": {
                    "name": "Systemic Expected Shortfall (SES)",
                    "description": "Expected capital shortfall in a systemic event",
                    "unit": "currency"
                },
                "marginal_expected_shortfall": {
                    "name": "Marginal Expected Shortfall (MES)",
                    "description": "Institution's contribution to systemic risk",
                    "unit": "percent"
                },
                "distress_insurance_premium": {
                    "name": "Distress Insurance Premium (DIP)",
                    "description": "Cost to insure against systemic distress",
                    "unit": "basis_points"
                }
            }
        },
        {
            "id": "market_based_indicators",
            "name": "Market-Based Risk Indicators",
            "description": "Market-implied systemic risk measures",
            "required_fields": [
                "cds_spreads",
                "equity_correlations",
                "implied_volatility",
                "systemic_risk_index"
            ]
        },
        {
            "id": "liquidity_risk",
            "name": "Systemic Liquidity Risk",
            "description": "Assessment of system-wide liquidity risk",
            "required_fields": [
                "market_liquidity_indicators",
                "funding_liquidity_indicators",
                "liquidity_mismatch_index",
                "fire_sale_risk"
            ]
        },
        {
            "id": "concentration_risk",
            "name": "Concentration and Common Exposure Risk",
            "description": "Analysis of concentration and common exposures",
            "required_fields": [
                "sectoral_concentration",
                "geographic_concentration",
                "asset_class_concentration",
                "counterparty_concentration"
            ]
        },
        {
            "id": "resilience_assessment",
            "name": "System Resilience Assessment",
            "description": "Evaluation of system's ability to absorb shocks",
            "required_fields": [
                "aggregate_capital_buffers",
                "aggregate_liquidity_buffers",
                "loss_absorption_capacity",
                "recovery_time_estimates"
            ]
        },
        {
            "id": "regulatory_framework",
            "name": "Regulatory and Supervisory Framework",
            "description": "Assessment of regulatory safeguards",
            "required_fields": [
                "macroprudential_measures",
                "resolution_framework",
                "crisis_management_protocols",
                "international_coordination"
            ]
        },
        {
            "id": "risk_mitigation",
            "name": "Risk Mitigation Recommendations",
            "description": "Recommended actions to reduce systemic risk",
            "required_fields": [
                "short_term_actions",
                "medium_term_actions",
                "long_term_structural_reforms",
                "monitoring_requirements"
            ]
        }
    ],
    "appendices": [
        {
            "id": "methodology",
            "name": "Analytical Methodology",
            "description": "Methods and models used in the assessment"
        },
        {
            "id": "data_sources",
            "name": "Data Sources and Quality",
            "description": "Description of data used and quality assessment"
        },
        {
            "id": "sensitivity",
            "name": "Sensitivity and Scenario Analysis",
            "description": "Alternative scenarios and sensitivity tests"
        },
        {
            "id": "glossary",
            "name": "Glossary of Terms",
            "description": "Definitions of technical terms and metrics"
        }
    ]
}


def get_template_by_type(report_type: str) -> Dict[str, Any]:
    """
    Get report template by type.

    Args:
        report_type: Type of report (basel_iii, stress_test, systemic_risk)

    Returns:
        Report template dictionary

    Raises:
        ValueError: If report type is not recognized
    """
    templates = {
        "basel_iii": BASEL_III_TEMPLATE,
        "stress_test": STRESS_TEST_TEMPLATE,
        "systemic_risk": SYSTEMIC_RISK_TEMPLATE,
    }

    if report_type not in templates:
        raise ValueError(
            f"Unknown report type: {report_type}. "
            f"Available types: {', '.join(templates.keys())}"
        )

    return templates[report_type]


def validate_report_data(report_type: str, data: Dict[str, Any]) -> List[str]:
    """
    Validate report data against template requirements.

    Args:
        report_type: Type of report
        data: Report data to validate

    Returns:
        List of validation errors (empty if valid)
    """
    template = get_template_by_type(report_type)
    errors = []

    for section in template["sections"]:
        section_id = section["id"]
        required_fields = section.get("required_fields", [])

        section_data = data.get(section_id, {})

        for field in required_fields:
            if field not in section_data:
                errors.append(
                    f"Missing required field '{field}' in section '{section_id}'"
                )

    return errors
