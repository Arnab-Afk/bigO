"""
PDF Report Generator for RUDRA Platform

Generates regulatory-grade PDF reports with network graphs, risk metrics,
and analysis for Basel III compliance, stress testing, and systemic risk assessment.
"""

import io
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import KeepTogether

from app.models.simulation import Simulation, SimulationResult
from app.models.institution import Institution
from app.reports.templates import BASEL_III_TEMPLATE, STRESS_TEST_TEMPLATE, SYSTEMIC_RISK_TEMPLATE

logger = logging.getLogger(__name__)


class RegulatoryReportGenerator:
    """
    Generates regulatory-compliant PDF reports for financial network simulations.

    Supports:
    - Basel III compliance reports
    - CCAR-style stress test reports
    - Systemic risk assessment reports
    - Individual bank analysis reports
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory to save generated PDFs (default: /tmp/rudra_reports)
        """
        self.output_dir = output_dir or Path("/tmp/rudra_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        # Page settings
        self.page_size = A4
        self.margin = 0.75 * inch

    def _setup_custom_styles(self) -> None:
        """Setup custom paragraph styles for reports."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))

        # Subsection header
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#3a7bc8'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        ))

        # Metric label
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#666666'),
            fontName='Helvetica-Bold'
        ))

        # Metric value
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#1a1a1a'),
            fontName='Helvetica'
        ))

        # Footer
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#888888'),
            alignment=TA_CENTER
        ))

    def generate_simulation_report(
        self,
        simulation: Simulation,
        results: List[SimulationResult],
        network_data: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate comprehensive simulation report.

        Args:
            simulation: Simulation model instance
            results: List of simulation results
            network_data: Optional network graph data for visualization

        Returns:
            Path to generated PDF file
        """
        logger.info(f"Generating simulation report for {simulation.id}")

        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_report_{simulation.id}_{timestamp}.pdf"
        output_path = self.output_dir / filename

        # Build report
        doc = BaseDocTemplate(
            str(output_path),
            pagesize=self.page_size,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin,
        )

        # Create frames and templates
        frame = Frame(
            self.margin,
            self.margin,
            self.page_size[0] - 2 * self.margin,
            self.page_size[1] - 2 * self.margin,
            id='normal'
        )

        template = PageTemplate(id='main', frames=[frame], onPage=self._add_page_number)
        doc.addPageTemplates([template])

        # Build content
        story = []

        # Title page
        story.extend(self._build_title_page(
            "Simulation Analysis Report",
            simulation.name,
            simulation.description
        ))
        story.append(PageBreak())

        # Executive summary
        story.extend(self._build_executive_summary(simulation, results))
        story.append(PageBreak())

        # Simulation metrics
        story.extend(self._build_simulation_metrics(results))
        story.append(PageBreak())

        # Network analysis
        if network_data:
            story.extend(self._build_network_analysis(network_data))
            story.append(PageBreak())

        # Risk analysis
        story.extend(self._build_risk_analysis(results))
        story.append(PageBreak())

        # Cascade analysis
        story.extend(self._build_cascade_analysis(results))

        # Build PDF
        doc.build(story)

        logger.info(f"Simulation report generated: {output_path}")
        return output_path

    def generate_bank_report(
        self,
        bank: Institution,
        simulation_data: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate individual bank analysis report.

        Args:
            bank: Institution model instance
            simulation_data: Optional simulation-specific data for the bank

        Returns:
            Path to generated PDF file
        """
        logger.info(f"Generating bank report for {bank.id}")

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"bank_report_{bank.external_id}_{timestamp}.pdf"
        output_path = self.output_dir / filename

        doc = BaseDocTemplate(
            str(output_path),
            pagesize=self.page_size,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin,
        )

        frame = Frame(
            self.margin,
            self.margin,
            self.page_size[0] - 2 * self.margin,
            self.page_size[1] - 2 * self.margin,
            id='normal'
        )

        template = PageTemplate(id='main', frames=[frame], onPage=self._add_page_number)
        doc.addPageTemplates([template])

        story = []

        # Title page
        story.extend(self._build_title_page(
            "Bank Analysis Report",
            bank.name,
            f"{bank.type.value.upper()} - {bank.tier.value.upper()}"
        ))
        story.append(PageBreak())

        # Bank profile
        story.extend(self._build_bank_profile(bank))
        story.append(PageBreak())

        # Basel III compliance (if applicable)
        if bank.type.value == "bank":
            story.extend(self._build_basel_compliance(bank, simulation_data))
            story.append(PageBreak())

        # Risk metrics
        if simulation_data:
            story.extend(self._build_bank_risk_metrics(bank, simulation_data))

        doc.build(story)

        logger.info(f"Bank report generated: {output_path}")
        return output_path

    def generate_stress_test_report(
        self,
        stress_test_data: Dict[str, Any],
        test_id: UUID
    ) -> Path:
        """
        Generate CCAR-style stress test report.

        Args:
            stress_test_data: Stress test results data
            test_id: Unique identifier for the stress test

        Returns:
            Path to generated PDF file
        """
        logger.info(f"Generating stress test report for {test_id}")

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"stress_test_report_{test_id}_{timestamp}.pdf"
        output_path = self.output_dir / filename

        doc = BaseDocTemplate(
            str(output_path),
            pagesize=self.page_size,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin,
        )

        frame = Frame(
            self.margin,
            self.margin,
            self.page_size[0] - 2 * self.margin,
            self.page_size[1] - 2 * self.margin,
            id='normal'
        )

        template = PageTemplate(id='main', frames=[frame], onPage=self._add_page_number)
        doc.addPageTemplates([template])

        story = []

        # Title page
        story.extend(self._build_title_page(
            "Stress Test Analysis Report",
            stress_test_data.get("name", "Stress Test"),
            stress_test_data.get("scenario_description", "")
        ))
        story.append(PageBreak())

        # Stress test overview
        story.extend(self._build_stress_test_overview(stress_test_data))
        story.append(PageBreak())

        # Scenario analysis
        story.extend(self._build_scenario_analysis(stress_test_data))
        story.append(PageBreak())

        # Results by institution
        story.extend(self._build_stress_test_results(stress_test_data))
        story.append(PageBreak())

        # Capital adequacy
        story.extend(self._build_capital_adequacy(stress_test_data))

        doc.build(story)

        logger.info(f"Stress test report generated: {output_path}")
        return output_path

    # --- Helper Methods for Building Report Sections ---

    def _build_title_page(
        self,
        title: str,
        subtitle: str,
        description: Optional[str] = None
    ) -> List:
        """Build title page content."""
        content = []

        content.append(Spacer(1, 2 * inch))
        content.append(Paragraph(title, self.styles['ReportTitle']))
        content.append(Spacer(1, 0.3 * inch))

        if subtitle:
            subtitle_style = ParagraphStyle(
                name='Subtitle',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#666666'),
                alignment=TA_CENTER
            )
            content.append(Paragraph(subtitle, subtitle_style))
            content.append(Spacer(1, 0.2 * inch))

        if description:
            desc_style = ParagraphStyle(
                name='Description',
                parent=self.styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#888888'),
                alignment=TA_CENTER
            )
            content.append(Paragraph(description, desc_style))

        content.append(Spacer(1, 1 * inch))

        # Metadata table
        metadata = [
            ['Generated:', datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")],
            ['Platform:', 'RUDRA - Resilient Unified Decision & Risk Analytics'],
        ]

        meta_table = Table(metadata, colWidths=[2*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1a1a1a')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        content.append(meta_table)

        return content

    def _build_executive_summary(
        self,
        simulation: Simulation,
        results: List[SimulationResult]
    ) -> List:
        """Build executive summary section."""
        content = []

        content.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        # Get primary result
        final_result = next((r for r in results if r.result_type == 'final_metrics'), results[0] if results else None)

        if not final_result:
            content.append(Paragraph("No results available.", self.styles['Normal']))
            return content

        # Key metrics grid
        metrics_data = [
            ['Metric', 'Value', 'Status'],
            [
                'Total Defaults',
                str(final_result.total_defaults),
                self._get_status_indicator(final_result.total_defaults, 'defaults')
            ],
            [
                'Survival Rate',
                f"{float(final_result.survival_rate) * 100:.2f}%",
                self._get_status_indicator(float(final_result.survival_rate), 'survival')
            ],
            [
                'Max Cascade Depth',
                str(final_result.max_cascade_depth),
                self._get_status_indicator(final_result.max_cascade_depth, 'cascade')
            ],
            [
                'Systemic Stress',
                f"{float(final_result.final_systemic_stress):.4f}",
                self._get_status_indicator(float(final_result.final_systemic_stress), 'stress')
            ],
            [
                'Total System Loss',
                f"${float(final_result.total_system_loss):,.2f}",
                self._get_status_indicator(float(final_result.total_system_loss), 'loss')
            ],
        ]

        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 11),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 10),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))

        content.append(metrics_table)
        content.append(Spacer(1, 0.3 * inch))

        # Summary text
        summary_text = f"""
        The simulation '{simulation.name}' completed in {simulation.duration_seconds or 0:.2f} seconds,
        executing {simulation.current_timestep} of {simulation.total_timesteps} planned timesteps.
        The analysis revealed {final_result.total_defaults} institutional defaults with a survival rate of
        {float(final_result.survival_rate) * 100:.2f}%. The maximum cascade depth observed was
        {final_result.max_cascade_depth} levels, with a final systemic stress level of
        {float(final_result.final_systemic_stress):.4f}.
        """

        content.append(Paragraph(summary_text, self.styles['Normal']))

        return content

    def _build_simulation_metrics(self, results: List[SimulationResult]) -> List:
        """Build detailed simulation metrics section."""
        content = []

        content.append(Paragraph("Simulation Metrics", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        final_result = next((r for r in results if r.result_type == 'final_metrics'), results[0] if results else None)

        if not final_result:
            return content

        # Time series chart
        if final_result.timeline_data:
            chart_img = self._create_timeline_chart(final_result.timeline_data)
            if chart_img:
                content.append(Paragraph("Simulation Timeline", self.styles['SubsectionHeader']))
                content.append(chart_img)
                content.append(Spacer(1, 0.2 * inch))

        # Detailed metrics from metrics_data
        if final_result.metrics_data:
            content.append(Paragraph("Detailed Metrics", self.styles['SubsectionHeader']))
            metrics_text = self._format_metrics_dict(final_result.metrics_data)
            content.append(Paragraph(metrics_text, self.styles['Normal']))

        return content

    def _build_network_analysis(self, network_data: Dict[str, Any]) -> List:
        """Build network analysis section with visualizations."""
        content = []

        content.append(Paragraph("Network Analysis", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        # Network statistics
        content.append(Paragraph("Network Statistics", self.styles['SubsectionHeader']))

        stats = network_data.get('statistics', {})
        stats_data = [
            ['Metric', 'Value'],
            ['Total Nodes', str(stats.get('total_nodes', 0))],
            ['Total Edges', str(stats.get('total_edges', 0))],
            ['Network Density', f"{stats.get('density', 0):.4f}"],
            ['Average Degree', f"{stats.get('avg_degree', 0):.2f}"],
            ['Clustering Coefficient', f"{stats.get('clustering', 0):.4f}"],
        ]

        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 11),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 10),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))

        content.append(stats_table)
        content.append(Spacer(1, 0.2 * inch))

        # Network visualization
        if 'nodes' in network_data and 'edges' in network_data:
            content.append(Paragraph("Network Visualization", self.styles['SubsectionHeader']))
            network_img = self._create_network_graph(network_data)
            if network_img:
                content.append(network_img)

        return content

    def _build_risk_analysis(self, results: List[SimulationResult]) -> List:
        """Build risk analysis section."""
        content = []

        content.append(Paragraph("Risk Analysis", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        final_result = next((r for r in results if r.result_type == 'final_metrics'), results[0] if results else None)

        if not final_result:
            return content

        # Risk assessment
        content.append(Paragraph("Systemic Risk Assessment", self.styles['SubsectionHeader']))

        risk_level = self._assess_risk_level(float(final_result.final_systemic_stress))
        risk_text = f"""
        <b>Risk Level:</b> {risk_level}<br/>
        <b>Systemic Stress:</b> {float(final_result.final_systemic_stress):.4f}<br/>
        <b>Total System Loss:</b> ${float(final_result.total_system_loss):,.2f}<br/>
        <b>Institutions at Risk:</b> {final_result.total_defaults}<br/>
        """

        content.append(Paragraph(risk_text, self.styles['Normal']))
        content.append(Spacer(1, 0.2 * inch))

        # Risk distribution chart
        if final_result.metrics_data and 'risk_distribution' in final_result.metrics_data:
            chart_img = self._create_risk_distribution_chart(final_result.metrics_data['risk_distribution'])
            if chart_img:
                content.append(chart_img)

        return content

    def _build_cascade_analysis(self, results: List[SimulationResult]) -> List:
        """Build cascade analysis section."""
        content = []

        content.append(Paragraph("Cascade Analysis", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        final_result = next((r for r in results if r.result_type == 'final_metrics'), results[0] if results else None)

        if not final_result or not final_result.cascade_data:
            content.append(Paragraph("No cascade data available.", self.styles['Normal']))
            return content

        cascade = final_result.cascade_data

        # Cascade overview
        cascade_text = f"""
        <b>Maximum Cascade Depth:</b> {final_result.max_cascade_depth}<br/>
        <b>Time to First Default:</b> {final_result.time_to_first_default if final_result.time_to_first_default != -1 else 'No defaults'}<br/>
        <b>Total Defaults:</b> {final_result.total_defaults}<br/>
        """

        content.append(Paragraph(cascade_text, self.styles['Normal']))
        content.append(Spacer(1, 0.2 * inch))

        # Cascade paths
        if 'cascade_paths' in cascade:
            content.append(Paragraph("Cascade Paths", self.styles['SubsectionHeader']))
            paths_text = self._format_cascade_paths(cascade['cascade_paths'])
            content.append(Paragraph(paths_text, self.styles['Normal']))

        return content

    def _build_bank_profile(self, bank: Institution) -> List:
        """Build bank profile section."""
        content = []

        content.append(Paragraph("Bank Profile", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        profile_data = [
            ['Field', 'Value'],
            ['Institution Name', bank.name],
            ['External ID', bank.external_id],
            ['Short Name', bank.short_name or 'N/A'],
            ['Type', bank.type.value.upper()],
            ['Systemic Tier', bank.tier.value.upper()],
            ['Jurisdiction', bank.jurisdiction or 'N/A'],
            ['Region', bank.region or 'N/A'],
            ['Status', 'Active' if bank.is_active else 'Inactive'],
        ]

        profile_table = Table(profile_data, colWidths=[2.5*inch, 4*inch])
        profile_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 11),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 10),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))

        content.append(profile_table)

        if bank.description:
            content.append(Spacer(1, 0.2 * inch))
            content.append(Paragraph("Description", self.styles['SubsectionHeader']))
            content.append(Paragraph(bank.description, self.styles['Normal']))

        return content

    def _build_basel_compliance(
        self,
        bank: Institution,
        simulation_data: Optional[Dict[str, Any]]
    ) -> List:
        """Build Basel III compliance section."""
        content = []

        content.append(Paragraph("Basel III Compliance", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        if not simulation_data or 'basel_metrics' not in simulation_data:
            content.append(Paragraph("Basel III metrics not available.", self.styles['Normal']))
            return content

        basel = simulation_data['basel_metrics']

        # Capital ratios
        content.append(Paragraph("Capital Adequacy Ratios", self.styles['SubsectionHeader']))

        ratios_data = [
            ['Ratio', 'Value', 'Minimum', 'Status'],
            [
                'CET1 Ratio',
                f"{basel.get('cet1_ratio', 0):.2f}%",
                '4.5%',
                'Pass' if basel.get('cet1_ratio', 0) >= 4.5 else 'Fail'
            ],
            [
                'Tier 1 Ratio',
                f"{basel.get('tier1_ratio', 0):.2f}%",
                '6.0%',
                'Pass' if basel.get('tier1_ratio', 0) >= 6.0 else 'Fail'
            ],
            [
                'Total Capital Ratio',
                f"{basel.get('total_capital_ratio', 0):.2f}%",
                '8.0%',
                'Pass' if basel.get('total_capital_ratio', 0) >= 8.0 else 'Fail'
            ],
        ]

        ratios_table = Table(ratios_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        ratios_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 11),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 10),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))

        content.append(ratios_table)

        return content

    def _build_bank_risk_metrics(
        self,
        bank: Institution,
        simulation_data: Dict[str, Any]
    ) -> List:
        """Build bank-specific risk metrics."""
        content = []

        content.append(Paragraph("Risk Metrics", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        risk_metrics = simulation_data.get('risk_metrics', {})

        metrics_data = [
            ['Metric', 'Value'],
            ['Credit Risk', f"{risk_metrics.get('credit_risk', 0):.4f}"],
            ['Liquidity Risk', f"{risk_metrics.get('liquidity_risk', 0):.4f}"],
            ['Market Risk', f"{risk_metrics.get('market_risk', 0):.4f}"],
            ['Operational Risk', f"{risk_metrics.get('operational_risk', 0):.4f}"],
            ['Systemic Risk', f"{risk_metrics.get('systemic_risk', 0):.4f}"],
        ]

        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 11),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 10),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))

        content.append(metrics_table)

        return content

    def _build_stress_test_overview(self, stress_test_data: Dict[str, Any]) -> List:
        """Build stress test overview section."""
        content = []

        content.append(Paragraph("Stress Test Overview", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        overview = stress_test_data.get('overview', {})

        overview_text = f"""
        <b>Test Name:</b> {stress_test_data.get('name', 'N/A')}<br/>
        <b>Scenario:</b> {stress_test_data.get('scenario', 'N/A')}<br/>
        <b>Severity:</b> {overview.get('severity', 'N/A')}<br/>
        <b>Institutions Tested:</b> {overview.get('institutions_tested', 0)}<br/>
        <b>Test Duration:</b> {overview.get('duration', 'N/A')}<br/>
        """

        content.append(Paragraph(overview_text, self.styles['Normal']))

        return content

    def _build_scenario_analysis(self, stress_test_data: Dict[str, Any]) -> List:
        """Build scenario analysis section."""
        content = []

        content.append(Paragraph("Scenario Analysis", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        scenarios = stress_test_data.get('scenarios', [])

        for scenario in scenarios:
            content.append(Paragraph(scenario.get('name', 'Scenario'), self.styles['SubsectionHeader']))
            content.append(Paragraph(scenario.get('description', ''), self.styles['Normal']))
            content.append(Spacer(1, 0.1 * inch))

        return content

    def _build_stress_test_results(self, stress_test_data: Dict[str, Any]) -> List:
        """Build stress test results section."""
        content = []

        content.append(Paragraph("Test Results by Institution", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        results = stress_test_data.get('institution_results', [])

        if not results:
            content.append(Paragraph("No results available.", self.styles['Normal']))
            return content

        # Results table
        headers = ['Institution', 'Pre-Stress CET1', 'Post-Stress CET1', 'Change', 'Status']
        table_data = [headers]

        for result in results[:20]:  # Limit to 20 institutions
            table_data.append([
                result.get('institution', 'N/A'),
                f"{result.get('pre_cet1', 0):.2f}%",
                f"{result.get('post_cet1', 0):.2f}%",
                f"{result.get('change', 0):.2f}%",
                result.get('status', 'N/A')
            ])

        results_table = Table(table_data, colWidths=[2*inch, 1.3*inch, 1.3*inch, 1.2*inch, 1*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))

        content.append(results_table)

        return content

    def _build_capital_adequacy(self, stress_test_data: Dict[str, Any]) -> List:
        """Build capital adequacy section."""
        content = []

        content.append(Paragraph("Capital Adequacy Assessment", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2 * inch))

        adequacy = stress_test_data.get('capital_adequacy', {})

        adequacy_text = f"""
        <b>Institutions Passing:</b> {adequacy.get('passing', 0)}<br/>
        <b>Institutions Failing:</b> {adequacy.get('failing', 0)}<br/>
        <b>Pass Rate:</b> {adequacy.get('pass_rate', 0):.2f}%<br/>
        <b>Average CET1 Decline:</b> {adequacy.get('avg_decline', 0):.2f}%<br/>
        """

        content.append(Paragraph(adequacy_text, self.styles['Normal']))

        return content

    # --- Visualization Helper Methods ---

    def _create_timeline_chart(self, timeline_data: Dict[str, Any]) -> Optional[Image]:
        """Create timeline chart from simulation data."""
        try:
            fig, ax = plt.subplots(figsize=(8, 5))

            timesteps = timeline_data.get('timesteps', [])
            if not timesteps:
                return None

            x = list(range(len(timesteps)))
            defaults = [t.get('defaults', 0) for t in timesteps]
            stress = [t.get('stress_level', 0) for t in timesteps]

            ax.plot(x, defaults, label='Defaults', color='#d32f2f', linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(x, stress, label='Stress Level', color='#1976d2', linewidth=2)

            ax.set_xlabel('Timestep')
            ax.set_ylabel('Defaults', color='#d32f2f')
            ax2.set_ylabel('Stress Level', color='#1976d2')
            ax.tick_params(axis='y', labelcolor='#d32f2f')
            ax2.tick_params(axis='y', labelcolor='#1976d2')
            ax.grid(True, alpha=0.3)

            plt.title('Simulation Timeline')
            plt.tight_layout()

            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()

            return Image(img_buffer, width=6*inch, height=3.75*inch)

        except Exception as e:
            logger.error(f"Error creating timeline chart: {e}")
            return None

    def _create_network_graph(self, network_data: Dict[str, Any]) -> Optional[Image]:
        """Create network visualization."""
        try:
            G = nx.DiGraph()

            # Add nodes
            for node in network_data.get('nodes', []):
                G.add_node(node['id'], **node)

            # Add edges
            for edge in network_data.get('edges', []):
                G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1))

            # Limit visualization to reasonable size
            if len(G.nodes()) > 100:
                # Sample largest connected component
                largest_cc = max(nx.weakly_connected_components(G), key=len)
                G = G.subgraph(list(largest_cc)[:100])

            fig, ax = plt.subplots(figsize=(8, 8))

            pos = nx.spring_layout(G, k=0.5, iterations=50)

            # Draw network
            nx.draw_networkx_nodes(G, pos, node_color='#2c5aa0', node_size=300, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=6, font_color='white', ax=ax)

            ax.set_title('Network Structure')
            ax.axis('off')
            plt.tight_layout()

            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()

            return Image(img_buffer, width=6*inch, height=6*inch)

        except Exception as e:
            logger.error(f"Error creating network graph: {e}")
            return None

    def _create_risk_distribution_chart(self, risk_data: Dict[str, Any]) -> Optional[Image]:
        """Create risk distribution chart."""
        try:
            fig, ax = plt.subplots(figsize=(8, 5))

            categories = list(risk_data.keys())
            values = list(risk_data.values())

            colors_palette = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c', '#1976d2']

            ax.bar(categories, values, color=colors_palette[:len(categories)])
            ax.set_ylabel('Risk Level')
            ax.set_title('Risk Distribution')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()

            return Image(img_buffer, width=6*inch, height=3.75*inch)

        except Exception as e:
            logger.error(f"Error creating risk distribution chart: {e}")
            return None

    # --- Utility Helper Methods ---

    def _add_page_number(self, canvas: canvas.Canvas, doc: BaseDocTemplate) -> None:
        """Add page number to footer."""
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#888888'))
        canvas.drawCentredString(
            self.page_size[0] / 2,
            0.5 * inch,
            text
        )
        canvas.restoreState()

    def _get_status_indicator(self, value: float, metric_type: str) -> str:
        """Get status indicator for a metric."""
        if metric_type == 'defaults':
            return 'Low' if value < 5 else 'Medium' if value < 20 else 'High'
        elif metric_type == 'survival':
            return 'High' if value > 0.9 else 'Medium' if value > 0.7 else 'Low'
        elif metric_type == 'cascade':
            return 'Low' if value < 3 else 'Medium' if value < 6 else 'High'
        elif metric_type == 'stress':
            return 'Low' if value < 0.3 else 'Medium' if value < 0.6 else 'High'
        elif metric_type == 'loss':
            return 'Low' if value < 1000000 else 'Medium' if value < 10000000 else 'High'
        return 'N/A'

    def _assess_risk_level(self, stress: float) -> str:
        """Assess overall risk level."""
        if stress < 0.3:
            return 'LOW RISK'
        elif stress < 0.6:
            return 'MEDIUM RISK'
        else:
            return 'HIGH RISK'

    def _format_metrics_dict(self, metrics: Dict[str, Any], indent: int = 0) -> str:
        """Format metrics dictionary as HTML."""
        lines = []
        for key, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"{'&nbsp;' * (indent * 4)}<b>{key}:</b>")
                lines.append(self._format_metrics_dict(value, indent + 1))
            else:
                lines.append(f"{'&nbsp;' * (indent * 4)}<b>{key}:</b> {value}")
        return '<br/>'.join(lines)

    def _format_cascade_paths(self, paths: List[Dict[str, Any]]) -> str:
        """Format cascade paths as HTML."""
        lines = []
        for i, path in enumerate(paths[:10], 1):  # Limit to 10 paths
            nodes = ' → '.join(path.get('nodes', []))
            lines.append(f"<b>Path {i}:</b> {nodes}")
        return '<br/>'.join(lines)
