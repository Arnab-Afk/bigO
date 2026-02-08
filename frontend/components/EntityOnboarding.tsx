"use client";

import { useState } from "react";
import { Building2, Shield, TrendingUp, Landmark, ChevronRight } from "lucide-react";
import { EntityType, BankPolicies, CCPPolicies, RegulatorPolicies, SectorPolicies, UserEntity } from "@/types/user";

interface EntityOnboardingProps {
    onComplete: (entity: UserEntity) => void;
}

const ENTITY_TYPES = [
    {
        type: "bank" as EntityType,
        icon: Building2,
        title: "Commercial Bank",
        description: "Manage lending, capital ratios, and liquidity buffers. Balance risk and return.",
        color: "bg-blue-500",
        hoverColor: "hover:bg-blue-600",
    },
    {
        type: "clearing_house" as EntityType,
        icon: Shield,
        title: "Clearing House (CCP)",
        description: "Set margin requirements and haircuts. Manage systemic stability and default risk.",
        color: "bg-purple-500",
        hoverColor: "hover:bg-purple-600",
    },
    {
        type: "regulator" as EntityType,
        icon: Landmark,
        title: "Regulator",
        description: "Set monetary policy and regulations. Intervene during crises to maintain stability.",
        color: "bg-amber-500",
        hoverColor: "hover:bg-amber-600",
    },
    {
        type: "sector" as EntityType,
        icon: TrendingUp,
        title: "Economic Sector",
        description: "Represent a sector like real estate or manufacturing. Influence borrowing institutions.",
        color: "bg-emerald-500",
        hoverColor: "hover:bg-emerald-600",
    },
];

const DEFAULT_POLICIES = {
    bank: {
        riskAppetite: 0.6,
        minCapitalRatio: 11.5,
        liquidityBuffer: 15,
        maxExposurePerCounterparty: 25,
        npaThreshold: 8,
        autoLendingEnabled: true,
    } as BankPolicies,
    clearing_house: {
        initialMargin: 10,
        haircut: 5,
        defaultFundSize: 5000000,
        stressTestMultiplier: 1.5,
        autoMarginAdjustment: true,
        marginAdjustmentTrigger: 0.3,
    } as CCPPolicies,
    regulator: {
        baseRepoRate: 6.5,
        minimumCRAR: 9,
        crisisInterventionThreshold: 0.6,
        liquidityInjectionAmount: 10000000,
        autoInterventionEnabled: true,
    } as RegulatorPolicies,
    sector: {
        economicHealth: 0.8,
        debtLoad: 45,
        volatility: 0.2,
    } as SectorPolicies,
};

export default function EntityOnboarding({ onComplete }: EntityOnboardingProps) {
    const [step, setStep] = useState<"select" | "configure">("select");
    const [selectedType, setSelectedType] = useState<EntityType | null>(null);
    const [entityName, setEntityName] = useState("");
    const [policies, setPolicies] = useState<any>(null);

    const handleSelectType = (type: EntityType) => {
        setSelectedType(type);
        setPolicies(DEFAULT_POLICIES[type]);
        setEntityName(
            type === "bank"
                ? "My Bank"
                : type === "clearing_house"
                    ? "My CCP"
                    : type === "regulator"
                        ? "My Central Bank"
                        : "My Sector"
        );
        setStep("configure");
    };

    const handlePolicyChange = (key: string, value: any) => {
        setPolicies((prev: any) => ({ ...prev, [key]: value }));
    };

    const handleComplete = () => {
        if (!selectedType || !entityName || !policies) return;

        const entity: UserEntity = {
            id: `USER_${selectedType.toUpperCase()}`,
            type: selectedType,
            name: entityName,
            policies,
            customRules: [],
        };

        onComplete(entity);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center p-6">
            <div className="w-full max-w-6xl">
                {/* Header */}
                <div className="text-center mb-12">
                    <h1 className="text-4xl font-bold text-slate-900 mb-3">
                        Financial Network Simulator
                    </h1>
                    <p className="text-lg text-slate-600 max-w-2xl mx-auto">
                        Design your entity and explore how your decisions propagate through the financial system
                    </p>
                </div>

                {step === "select" && (
                    <div>
                        <h2 className="text-2xl font-bold text-slate-800 mb-6 text-center">
                            Choose Your Entity Type
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {ENTITY_TYPES.map((entity) => {
                                const Icon = entity.icon;
                                return (
                                    <button
                                        key={entity.type}
                                        onClick={() => handleSelectType(entity.type)}
                                        className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 border-2 border-transparent hover:border-slate-300 group text-left"
                                    >
                                        <div className={`w-16 h-16 ${entity.color} rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                                            <Icon className="w-8 h-8 text-white" />
                                        </div>
                                        <h3 className="text-2xl font-bold text-slate-900 mb-2">
                                            {entity.title}
                                        </h3>
                                        <p className="text-slate-600 leading-relaxed">
                                            {entity.description}
                                        </p>
                                        <div className="mt-4 flex items-center text-slate-500 group-hover:text-slate-700 transition-colors">
                                            <span className="text-sm font-medium">Get Started</span>
                                            <ChevronRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                )}

                {step === "configure" && selectedType && (
                    <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-3xl mx-auto">
                        <button
                            onClick={() => setStep("select")}
                            className="text-slate-600 hover:text-slate-900 mb-6 flex items-center text-sm font-medium"
                        >
                            ← Back to entity selection
                        </button>

                        <h2 className="text-2xl font-bold text-slate-900 mb-6">
                            Configure Your {ENTITY_TYPES.find((e) => e.type === selectedType)?.title}
                        </h2>

                        {/* Entity Name */}
                        <div className="mb-8">
                            <label className="block text-sm font-semibold text-slate-700 mb-2">
                                Entity Name
                            </label>
                            <input
                                type="text"
                                value={entityName}
                                onChange={(e) => setEntityName(e.target.value)}
                                className="w-full px-4 py-3 border-2 border-slate-200 rounded-lg focus:border-blue-500 focus:ring focus:ring-blue-200 outline-none text-slate-900"
                                placeholder="Enter a name for your entity"
                            />
                        </div>

                        {/* Policy Configuration */}
                        <h3 className="text-lg font-bold text-slate-800 mb-4">Initial Policies</h3>
                        <div className="space-y-4 mb-8">
                            {selectedType === "bank" && (
                                <>
                                    <PolicySlider
                                        label="Risk Appetite"
                                        value={(policies as BankPolicies).riskAppetite}
                                        onChange={(v) => handlePolicyChange("riskAppetite", v)}
                                        min={0}
                                        max={1}
                                        step={0.1}
                                        suffix=""
                                        description="How aggressively to pursue high-risk, high-return opportunities"
                                    />
                                    <PolicySlider
                                        label="Min Capital Ratio"
                                        value={(policies as BankPolicies).minCapitalRatio}
                                        onChange={(v) => handlePolicyChange("minCapitalRatio", v)}
                                        min={9}
                                        max={20}
                                        step={0.5}
                                        suffix="%"
                                        description="Minimum CRAR before restricting lending"
                                    />
                                    <PolicySlider
                                        label="Liquidity Buffer"
                                        value={(policies as BankPolicies).liquidityBuffer}
                                        onChange={(v) => handlePolicyChange("liquidityBuffer", v)}
                                        min={5}
                                        max={30}
                                        step={1}
                                        suffix="%"
                                        description="Percentage of assets held as liquid reserves"
                                    />
                                    <PolicySlider
                                        label="Max Exposure per Counterparty"
                                        value={(policies as BankPolicies).maxExposurePerCounterparty}
                                        onChange={(v) => handlePolicyChange("maxExposurePerCounterparty", v)}
                                        min={10}
                                        max={50}
                                        step={5}
                                        suffix="%"
                                        description="Maximum lending to any single institution (% of capital)"
                                    />
                                </>
                            )}

                            {selectedType === "clearing_house" && (
                                <>
                                    <PolicySlider
                                        label="Initial Margin"
                                        value={(policies as CCPPolicies).initialMargin}
                                        onChange={(v) => handlePolicyChange("initialMargin", v)}
                                        min={5}
                                        max={30}
                                        step={1}
                                        suffix="%"
                                        description="Collateral required for clearing trades"
                                    />
                                    <PolicySlider
                                        label="Haircut Rate"
                                        value={(policies as CCPPolicies).haircut}
                                        onChange={(v) => handlePolicyChange("haircut", v)}
                                        min={0}
                                        max={20}
                                        step={1}
                                        suffix="%"
                                        description="Discount on collateral value"
                                    />
                                    <PolicySlider
                                        label="Stress Test Multiplier"
                                        value={(policies as CCPPolicies).stressTestMultiplier}
                                        onChange={(v) => handlePolicyChange("stressTestMultiplier", v)}
                                        min={1}
                                        max={3}
                                        step={0.1}
                                        suffix="x"
                                        description="How much to buffer against worst-case scenarios"
                                    />
                                </>
                            )}

                            {selectedType === "regulator" && (
                                <>
                                    <PolicySlider
                                        label="Base Repo Rate"
                                        value={(policies as RegulatorPolicies).baseRepoRate}
                                        onChange={(v) => handlePolicyChange("baseRepoRate", v)}
                                        min={2}
                                        max={15}
                                        step={0.25}
                                        suffix="%"
                                        description="Central bank lending rate to commercial banks"
                                    />
                                    <PolicySlider
                                        label="Minimum CRAR Requirement"
                                        value={(policies as RegulatorPolicies).minimumCRAR}
                                        onChange={(v) => handlePolicyChange("minimumCRAR", v)}
                                        min={8}
                                        max={15}
                                        step={0.5}
                                        suffix="%"
                                        description="Regulatory minimum capital adequacy ratio"
                                    />
                                    <PolicySlider
                                        label="Crisis Intervention Threshold"
                                        value={(policies as RegulatorPolicies).crisisInterventionThreshold}
                                        onChange={(v) => handlePolicyChange("crisisInterventionThreshold", v)}
                                        min={0.3}
                                        max={0.9}
                                        step={0.05}
                                        suffix=""
                                        description="System health level below which to intervene"
                                    />
                                </>
                            )}

                            {selectedType === "sector" && (
                                <>
                                    <PolicySlider
                                        label="Economic Health"
                                        value={(policies as SectorPolicies).economicHealth}
                                        onChange={(v) => handlePolicyChange("economicHealth", v)}
                                        min={0.1}
                                        max={1}
                                        step={0.05}
                                        suffix=""
                                        description="Sector performance level (affects borrowers)"
                                    />
                                    <PolicySlider
                                        label="Debt Load"
                                        value={(policies as SectorPolicies).debtLoad}
                                        onChange={(v) => handlePolicyChange("debtLoad", v)}
                                        min={20}
                                        max={80}
                                        step={5}
                                        suffix="%"
                                        description="Sector indebtedness relative to GDP"
                                    />
                                    <PolicySlider
                                        label="Volatility"
                                        value={(policies as SectorPolicies).volatility}
                                        onChange={(v) => handlePolicyChange("volatility", v)}
                                        min={0}
                                        max={1}
                                        step={0.05}
                                        suffix=""
                                        description="How unpredictable sector performance is"
                                    />
                                </>
                            )}
                        </div>

                        <button
                            onClick={handleComplete}
                            disabled={!entityName.trim()}
                            className="w-full bg-gradient-to-r from-blue-600 to-blue-500 text-white py-4 rounded-lg font-bold text-lg hover:from-blue-700 hover:to-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl"
                        >
                            Start Simulation →
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}

interface PolicySliderProps {
    label: string;
    value: number;
    onChange: (value: number) => void;
    min: number;
    max: number;
    step: number;
    suffix: string;
    description: string;
}

function PolicySlider({ label, value, onChange, min, max, step, suffix, description }: PolicySliderProps) {
    return (
        <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
            <div className="flex justify-between items-center mb-2">
                <label className="font-semibold text-slate-800">{label}</label>
                <span className="font-mono text-lg font-bold text-blue-600">
                    {suffix === "" ? value.toFixed(2) : `${value}${suffix}`}
                </span>
            </div>
            <input
                type="range"
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
                min={min}
                max={max}
                step={step}
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
            <p className="text-xs text-slate-600 mt-2">{description}</p>
        </div>
    );
}
