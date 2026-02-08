/**
 * User and Entity Type Definitions
 */

export type EntityType = "bank" | "clearing_house" | "regulator" | "sector";

export interface PolicyRule {
    id: string;
    name: string;
    condition: string;
    action: string;
    enabled: boolean;
}

export interface BankPolicies {
    riskAppetite: number; // 0-1 scale
    minCapitalRatio: number; // percentage
    liquidityBuffer: number; // percentage
    maxExposurePerCounterparty: number; // percentage of capital
    npaThreshold: number; // percentage
    autoLendingEnabled: boolean;
}

export interface CCPPolicies {
    initialMargin: number; // percentage
    haircut: number; // percentage
    defaultFundSize: number; // absolute value
    stressTestMultiplier: number; // multiplier
    autoMarginAdjustment: boolean;
    marginAdjustmentTrigger: number; // volatility threshold
}

export interface RegulatorPolicies {
    baseRepoRate: number; // percentage
    minimumCRAR: number; // percentage
    crisisInterventionThreshold: number; // system health threshold
    liquidityInjectionAmount: number; // absolute value
    autoInterventionEnabled: boolean;
}

export interface SectorPolicies {
    economicHealth: number; // 0-1 scale
    debtLoad: number; // percentage
    volatility: number; // 0-1 scale
}

export type EntityPolicies = BankPolicies | CCPPolicies | RegulatorPolicies | SectorPolicies;

export interface UserEntity {
    id: string;
    type: EntityType;
    name: string;
    policies: EntityPolicies;
    customRules: PolicyRule[];
}

export interface UserSession {
    userId: string;
    entity: UserEntity;
    simulationId: string | null;
    createdAt: Date;
}
