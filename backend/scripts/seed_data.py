"""
Sample data seeding script for development
"""

import asyncio
import random
from datetime import datetime, timedelta, date
from decimal import Decimal
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_factory, init_db
from app.models.institution import Institution, InstitutionType, SystemicTier
from app.models.institution_state import InstitutionState
from app.models.exposure import Exposure, ExposureType
from app.models.scenario import Scenario, Shock, ShockType


# Sample institution data
SAMPLE_INSTITUTIONS = [
    # G-SIBs
    {"name": "Global Bank of New York", "type": InstitutionType.BANK, "tier": SystemicTier.G_SIB, "jurisdiction": "US"},
    {"name": "London International Bank", "type": InstitutionType.BANK, "tier": SystemicTier.G_SIB, "jurisdiction": "GB"},
    {"name": "Deutsche Financial AG", "type": InstitutionType.BANK, "tier": SystemicTier.G_SIB, "jurisdiction": "DE"},
    {"name": "Tokyo Metropolitan Bank", "type": InstitutionType.BANK, "tier": SystemicTier.G_SIB, "jurisdiction": "JP"},
    
    # D-SIBs
    {"name": "American Regional Bank", "type": InstitutionType.BANK, "tier": SystemicTier.D_SIB, "jurisdiction": "US"},
    {"name": "European Commercial Bank", "type": InstitutionType.BANK, "tier": SystemicTier.D_SIB, "jurisdiction": "FR"},
    {"name": "Asian Pacific Bank", "type": InstitutionType.BANK, "tier": SystemicTier.D_SIB, "jurisdiction": "SG"},
    
    # CCPs
    {"name": "Global Clearing Corporation", "type": InstitutionType.CCP, "tier": SystemicTier.TIER_1, "jurisdiction": "US"},
    {"name": "European Clearing House", "type": InstitutionType.CCP, "tier": SystemicTier.TIER_1, "jurisdiction": "DE"},
    
    # Exchanges
    {"name": "New York Securities Exchange", "type": InstitutionType.EXCHANGE, "tier": SystemicTier.TIER_1, "jurisdiction": "US"},
    {"name": "London Stock Exchange", "type": InstitutionType.EXCHANGE, "tier": SystemicTier.TIER_1, "jurisdiction": "GB"},
    
    # Tier 2 Banks
    {"name": "Midwest Community Bank", "type": InstitutionType.BANK, "tier": SystemicTier.TIER_2, "jurisdiction": "US"},
    {"name": "Nordic Investment Bank", "type": InstitutionType.BANK, "tier": SystemicTier.TIER_2, "jurisdiction": "NO"},
    {"name": "Swiss Private Bank", "type": InstitutionType.BANK, "tier": SystemicTier.TIER_2, "jurisdiction": "CH"},
    
    # Asset Managers
    {"name": "Global Asset Management", "type": InstitutionType.ASSET_MANAGER, "tier": SystemicTier.TIER_2, "jurisdiction": "US"},
    {"name": "European Investment Fund", "type": InstitutionType.ASSET_MANAGER, "tier": SystemicTier.TIER_2, "jurisdiction": "LU"},
    
    # Brokers
    {"name": "Prime Brokerage Corp", "type": InstitutionType.BROKER, "tier": SystemicTier.TIER_2, "jurisdiction": "US"},
    {"name": "Asian Trading Services", "type": InstitutionType.BROKER, "tier": SystemicTier.TIER_3, "jurisdiction": "HK"},
    
    # Tier 3 Banks
    {"name": "Local Savings Bank", "type": InstitutionType.BANK, "tier": SystemicTier.TIER_3, "jurisdiction": "US"},
    {"name": "Regional Credit Union", "type": InstitutionType.BANK, "tier": SystemicTier.TIER_3, "jurisdiction": "CA"},
]


async def seed_institutions(session: AsyncSession) -> list[Institution]:
    """Create sample institutions"""
    institutions = []
    
    for i, inst_data in enumerate(SAMPLE_INSTITUTIONS):
        institution = Institution(
            external_id=f"INST-{i+1:04d}",
            name=inst_data["name"],
            short_name=inst_data["name"].split()[0][:10],
            type=inst_data["type"],
            tier=inst_data["tier"],
            jurisdiction=inst_data["jurisdiction"],
            region="North America" if inst_data["jurisdiction"] in ["US", "CA"] else "Europe" if inst_data["jurisdiction"] in ["GB", "DE", "FR", "CH", "NO", "LU"] else "Asia Pacific",
            is_active=True,
            description=f"Sample {inst_data['type'].value} institution",
        )
        session.add(institution)
        institutions.append(institution)
    
    await session.flush()
    return institutions


async def seed_institution_states(session: AsyncSession, institutions: list[Institution]) -> None:
    """Create sample institution states"""
    now = datetime.utcnow()
    
    for inst in institutions:
        # Generate base values based on tier
        tier_multipliers = {
            SystemicTier.G_SIB: 1.0,
            SystemicTier.D_SIB: 0.8,
            SystemicTier.TIER_1: 0.6,
            SystemicTier.TIER_2: 0.4,
            SystemicTier.TIER_3: 0.2,
        }
        multiplier = tier_multipliers.get(inst.tier, 0.3)
        
        state = InstitutionState(
            institution_id=inst.id,
            timestamp=now,
            capital_ratio=Decimal(str(round(0.12 + random.uniform(-0.03, 0.05), 4))),
            leverage_ratio=Decimal(str(round(0.05 + random.uniform(-0.01, 0.02), 4))),
            total_capital=Decimal(str(round(multiplier * 100e9 * random.uniform(0.8, 1.2), 2))),
            risk_weighted_assets=Decimal(str(round(multiplier * 800e9 * random.uniform(0.8, 1.2), 2))),
            liquidity_coverage_ratio=Decimal(str(round(1.2 + random.uniform(-0.1, 0.3), 4))),
            net_stable_funding_ratio=Decimal(str(round(1.1 + random.uniform(-0.05, 0.2), 4))),
            liquidity_buffer=Decimal(str(round(0.8 + random.uniform(-0.1, 0.2), 4))),
            total_credit_exposure=Decimal(str(round(multiplier * 50e9 * random.uniform(0.5, 1.5), 2))),
            default_probability=Decimal(str(round(0.001 + random.uniform(0, 0.005), 6))),
            stress_level=Decimal(str(round(0.1 + random.uniform(0, 0.2), 4))),
            risk_score=Decimal(str(round(30 + random.uniform(-10, 20), 2))),
            risk_appetite=Decimal(str(round(0.5 + random.uniform(-0.2, 0.2), 4))),
            margin_sensitivity=Decimal(str(round(0.5 + random.uniform(-0.2, 0.2), 4))),
            source="seed_data",
        )
        session.add(state)
    
    await session.flush()


async def seed_exposures(session: AsyncSession, institutions: list[Institution]) -> None:
    """Create sample exposure relationships"""
    now = datetime.utcnow()
    
    # G-SIBs and D-SIBs are highly interconnected
    large_banks = [i for i in institutions if i.tier in [SystemicTier.G_SIB, SystemicTier.D_SIB]]
    ccps = [i for i in institutions if i.type == InstitutionType.CCP]
    other = [i for i in institutions if i not in large_banks and i not in ccps]
    
    # Create exposures between large banks
    for i, source in enumerate(large_banks):
        for j, target in enumerate(large_banks):
            if i != j and random.random() < 0.7:  # 70% connectivity
                exposure = Exposure(
                    source_institution_id=source.id,
                    target_institution_id=target.id,
                    exposure_type=random.choice([ExposureType.INTERBANK_LENDING, ExposureType.DERIVATIVES, ExposureType.REPO]),
                    gross_exposure=Decimal(str(round(random.uniform(1e9, 20e9), 2))),
                    net_exposure=Decimal(str(round(random.uniform(0.5e9, 15e9), 2))),
                    collateral_value=Decimal(str(round(random.uniform(0, 5e9), 2))),
                    collateral_haircut=Decimal(str(round(random.uniform(0.02, 0.15), 4))),
                    recovery_rate=Decimal(str(round(random.uniform(0.4, 0.7), 4))),
                    contagion_probability=Decimal(str(round(random.uniform(0.1, 0.4), 4))),
                    settlement_urgency=Decimal(str(round(random.uniform(0.3, 0.8), 4))),
                    effective_date=date.today() - timedelta(days=random.randint(30, 365)),
                    maturity_date=date.today() + timedelta(days=random.randint(90, 730)),
                    valid_from=now - timedelta(days=30),
                    currency="USD",
                )
                session.add(exposure)
    
    # Create exposures from large banks to CCPs
    for bank in large_banks:
        for ccp in ccps:
            if random.random() < 0.9:  # 90% connectivity to CCPs
                exposure = Exposure(
                    source_institution_id=bank.id,
                    target_institution_id=ccp.id,
                    exposure_type=ExposureType.CLEARING_MARGIN,
                    gross_exposure=Decimal(str(round(random.uniform(2e9, 30e9), 2))),
                    recovery_rate=Decimal("0.99"),
                    contagion_probability=Decimal(str(round(random.uniform(0.05, 0.15), 4))),
                    settlement_urgency=Decimal(str(round(random.uniform(0.7, 0.95), 4))),
                    effective_date=date.today() - timedelta(days=random.randint(30, 365)),
                    valid_from=now - timedelta(days=30),
                    currency="USD",
                )
                session.add(exposure)
    
    # Create some exposures from smaller institutions to large banks
    for small in other:
        for large in random.sample(large_banks, k=min(3, len(large_banks))):
            if random.random() < 0.5:
                exposure = Exposure(
                    source_institution_id=small.id,
                    target_institution_id=large.id,
                    exposure_type=random.choice([ExposureType.CREDIT_LINE, ExposureType.SETTLEMENT]),
                    gross_exposure=Decimal(str(round(random.uniform(0.1e9, 2e9), 2))),
                    recovery_rate=Decimal(str(round(random.uniform(0.5, 0.8), 4))),
                    contagion_probability=Decimal(str(round(random.uniform(0.05, 0.2), 4))),
                    settlement_urgency=Decimal(str(round(random.uniform(0.2, 0.6), 4))),
                    effective_date=date.today() - timedelta(days=random.randint(30, 365)),
                    valid_from=now - timedelta(days=30),
                    currency="USD",
                )
                session.add(exposure)
    
    await session.flush()


async def seed_scenarios(session: AsyncSession, institutions: list[Institution]) -> None:
    """Create sample simulation scenarios"""
    
    # Scenario 1: Single Institution Default
    scenario1 = Scenario(
        name="Single G-SIB Default",
        description="Simulates the default of a single G-SIB and its cascade effects",
        category="stress_test",
        is_template=True,
        num_timesteps=100,
        base_volatility=Decimal("0.20"),
        liquidity_premium=Decimal("0.03"),
    )
    session.add(scenario1)
    await session.flush()
    
    shock1 = Shock(
        scenario_id=scenario1.id,
        name="G-SIB Default",
        description="Sudden default of a major global bank",
        shock_type=ShockType.INSTITUTION_DEFAULT,
        target_type="institution",
        target_id=str(institutions[0].id),  # First G-SIB
        magnitude=Decimal("1.0"),
        duration=1,
        trigger_timestep=10,
    )
    session.add(shock1)
    
    # Scenario 2: Market Volatility Surge
    scenario2 = Scenario(
        name="Market Volatility Shock",
        description="Sudden spike in market volatility affecting all institutions",
        category="stress_test",
        is_template=True,
        num_timesteps=150,
        base_volatility=Decimal("0.15"),
    )
    session.add(scenario2)
    await session.flush()
    
    shock2 = Shock(
        scenario_id=scenario2.id,
        name="Volatility Spike",
        description="Market-wide volatility increase",
        shock_type=ShockType.MARKET_VOLATILITY,
        target_type="market",
        magnitude=Decimal("0.5"),
        duration=20,
        trigger_timestep=5,
    )
    session.add(shock2)
    
    # Scenario 3: Liquidity Crisis
    scenario3 = Scenario(
        name="Liquidity Crisis",
        description="System-wide liquidity freeze scenario",
        category="stress_test",
        is_template=True,
        num_timesteps=200,
        base_volatility=Decimal("0.25"),
        liquidity_premium=Decimal("0.10"),
    )
    session.add(scenario3)
    await session.flush()
    
    shock3a = Shock(
        scenario_id=scenario3.id,
        name="Initial Liquidity Freeze",
        shock_type=ShockType.LIQUIDITY_FREEZE,
        target_type="institution",
        target_id=str(institutions[1].id),
        magnitude=Decimal("0.7"),
        duration=10,
        trigger_timestep=5,
    )
    session.add(shock3a)
    
    shock3b = Shock(
        scenario_id=scenario3.id,
        name="Market Volatility Response",
        shock_type=ShockType.MARKET_VOLATILITY,
        target_type="market",
        magnitude=Decimal("0.3"),
        duration=30,
        trigger_timestep=8,
    )
    session.add(shock3b)
    
    await session.flush()


async def main():
    """Main seeding function"""
    print("Initializing database...")
    await init_db()
    
    print("Seeding data...")
    async with async_session_factory() as session:
        try:
            # Check if data already exists
            from sqlalchemy import select, func
            count = await session.scalar(select(func.count(Institution.id)))
            
            if count > 0:
                print(f"Database already has {count} institutions. Skipping seed.")
                return
            
            # Seed data
            print("Creating institutions...")
            institutions = await seed_institutions(session)
            
            print("Creating institution states...")
            await seed_institution_states(session, institutions)
            
            print("Creating exposures...")
            await seed_exposures(session, institutions)
            
            print("Creating scenarios...")
            await seed_scenarios(session, institutions)
            
            await session.commit()
            print(f"Successfully seeded {len(institutions)} institutions with states, exposures, and scenarios!")
            
        except Exception as e:
            await session.rollback()
            print(f"Error seeding data: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
