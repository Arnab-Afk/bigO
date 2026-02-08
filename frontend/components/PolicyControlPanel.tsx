"use client";

import { useState } from "react";
import { Settings, TrendingUp, AlertTriangle, Info, ChevronDown, ChevronUp } from "lucide-react";
import { UserEntity, EntityType, BankPolicies, CCPPolicies, RegulatorPolicies, SectorPolicies } from "@/types/user";

interface PolicyControlPanelProps {
    userEntity: UserEntity;
    onPolicyChange: (policies: any) => void;
    isSimulationRunning: boolean;
    currentHealth: number;
}

export default function PolicyControlPanel({
    userEntity,
    onPolicyChange,
    isSimulationRunning,
    currentHealth,
}: PolicyControlPanelProps) {
    const [isExpanded, setIsExpanded] = useState(true);
    const [activeTab, setActiveTab] = useState<"policies" | "rules">("policies");

    const handlePolicyChange = (key: string, value: any) => {
        const updatedPolicies = { ...userEntity.policies, [key]: value };
        onPolicyChange(updatedPolicies);
    };

    const getHealthColor = (health: number) => {
        if (health >= 0.7) return "text-green-600";
        if (health >= 0.4) return "text-amber-600";
        return "text-red-600";
    };

    return (
        <div className="bg-white rounded-xl shadow-lg border-2 border-slate-200">
            {/* Header */}
            <div className="p-4 border-b border-slate-200 bg-gradient-to-r from-blue-50 to-indigo-50">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                            <Settings className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h3 className="font-bold text-slate-900 text-lg">{userEntity.name}</h3>
                            <p className="text-xs text-slate-600 uppercase tracking-wide">
                                {userEntity.type.replace("_", " ")}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={() => setIsExpanded(!isExpanded)}
                        className="p-2 hover:bg-blue-100 rounded-lg transition-colors"
                    >
                        {isExpanded ? (
                            <ChevronUp className="w-5 h-5 text-slate-600" />
                        ) : (
                            <ChevronDown className="w-5 h-5 text-slate-600" />
                        )}
                    </button>
                </div>

                {/* Health Indicator */}
                <div className="mt-3 flex items-center gap-2">
                    <span className="text-xs font-medium text-slate-600">Health:</span>
                    <div className="flex-1 bg-slate-200 rounded-full h-2">
                        <div
                            className={`h-2 rounded-full transition-all ${currentHealth >= 0.7
                                ? "bg-green-500"
                                : currentHealth >= 0.4
                                    ? "bg-amber-500"
                                    : "bg-red-500"
                                }`}
                            style={{ width: `${currentHealth * 100}%` }}
                        />
                    </div>
                    <span className={`text-sm font-bold ${getHealthColor(currentHealth)}`}>
                        {(currentHealth * 100).toFixed(0)}%
                    </span>
                </div>
            </div>

            {isExpanded && (
                <div className="p-4">
                    {/* Tabs */}
                    <div className="flex gap-2 mb-4 border-b border-slate-200">
                        <button
                            onClick={() => setActiveTab("policies")}
                            className={`px-4 py-2 font-medium text-sm transition-colors ${activeTab === "policies"
                                ? "text-blue-600 border-b-2 border-blue-600"
                                : "text-slate-600 hover:text-slate-900"
                                }`}
                        >
                            Policies
                        </button>
                        <button
                            onClick={() => setActiveTab("rules")}
                            className={`px-4 py-2 font-medium text-sm transition-colors ${activeTab === "rules"
                                ? "text-blue-600 border-b-2 border-blue-600"
                                : "text-slate-600 hover:text-slate-900"
                                }`}
                        >
                            Custom Rules
                        </button>
                    </div>

                    {/* Policy Controls */}
                    {activeTab === "policies" && (
                        <div className="space-y-3 max-h-96 overflow-y-auto">
                            {userEntity.type === "bank" && (
                                <BankPolicyControls
                                    policies={userEntity.policies as BankPolicies}
                                    onChange={handlePolicyChange}
                                    disabled={isSimulationRunning}
                                />
                            )}
                            {userEntity.type === "clearing_house" && (
                                <CCPPolicyControls
                                    policies={userEntity.policies as CCPPolicies}
                                    onChange={handlePolicyChange}
                                    disabled={isSimulationRunning}
                                />
                            )}
                            {userEntity.type === "regulator" && (
                                <RegulatorPolicyControls
                                    policies={userEntity.policies as RegulatorPolicies}
                                    onChange={handlePolicyChange}
                                    disabled={isSimulationRunning}
                                />
                            )}
                            {userEntity.type === "sector" && (
                                <SectorPolicyControls
                                    policies={userEntity.policies as SectorPolicies}
                                    onChange={handlePolicyChange}
                                    disabled={isSimulationRunning}
                                />
                            )}

                            {isSimulationRunning && (
                                <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-start gap-2">
                                    <Info className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                                    <p className="text-xs text-amber-800">
                                        Policy changes during simulation will affect your entity in the next timestep.
                                    </p>
                                </div>
                            )}
                        </div>
                    )}

                    {activeTab === "rules" && (
                        <div className="space-y-3">
                            <p className="text-sm text-slate-600">
                                Custom game-theoretic rules will be available in a future update.
                            </p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

// Bank Policy Controls
function BankPolicyControls({
    policies,
    onChange,
    disabled,
}: {
    policies: BankPolicies;
    onChange: (key: string, value: any) => void;
    disabled: boolean;
}) {
    return (
        <>
            <PolicyControl
                label="Risk Appetite"
                value={policies.riskAppetite}
                onChange={(v) => onChange("riskAppetite", v)}
                min={0}
                max={1}
                step={0.05}
                suffix=""
                disabled={disabled}
                description="Willingness to take risk for higher returns"
            />
            <PolicyControl
                label="Min Capital Ratio"
                value={policies.minCapitalRatio}
                onChange={(v) => onChange("minCapitalRatio", v)}
                min={9}
                max={20}
                step={0.5}
                suffix="%"
                disabled={disabled}
                description="Minimum CRAR before restricting lending"
            />
            <PolicyControl
                label="Liquidity Buffer"
                value={policies.liquidityBuffer}
                onChange={(v) => onChange("liquidityBuffer", v)}
                min={5}
                max={30}
                step={1}
                suffix="%"
                disabled={disabled}
                description="Cash reserves as % of assets"
            />
            <PolicyControl
                label="Max Exposure/Counterparty"
                value={policies.maxExposurePerCounterparty}
                onChange={(v) => onChange("maxExposurePerCounterparty", v)}
                min={10}
                max={50}
                step={5}
                suffix="%"
                disabled={disabled}
                description="Max lending to single institution"
            />
        </>
    );
}

// CCP Policy Controls
function CCPPolicyControls({
    policies,
    onChange,
    disabled,
}: {
    policies: CCPPolicies;
    onChange: (key: string, value: any) => void;
    disabled: boolean;
}) {
    return (
        <>
            <PolicyControl
                label="Initial Margin"
                value={policies.initialMargin}
                onChange={(v) => onChange("initialMargin", v)}
                min={5}
                max={30}
                step={1}
                suffix="%"
                disabled={disabled}
                description="Collateral required for trades"
            />
            <PolicyControl
                label="Haircut Rate"
                value={policies.haircut}
                onChange={(v) => onChange("haircut", v)}
                min={0}
                max={20}
                step={1}
                suffix="%"
                disabled={disabled}
                description="Discount on collateral value"
            />
            <PolicyControl
                label="Stress Test Multiplier"
                value={policies.stressTestMultiplier}
                onChange={(v) => onChange("stressTestMultiplier", v)}
                min={1}
                max={3}
                step={0.1}
                suffix="x"
                disabled={disabled}
                description="Worst-case scenario buffer"
            />
        </>
    );
}

// Regulator Policy Controls
function RegulatorPolicyControls({
    policies,
    onChange,
    disabled,
}: {
    policies: RegulatorPolicies;
    onChange: (key: string, value: any) => void;
    disabled: boolean;
}) {
    return (
        <>
            <PolicyControl
                label="Base Repo Rate"
                value={policies.baseRepoRate}
                onChange={(v) => onChange("baseRepoRate", v)}
                min={2}
                max={15}
                step={0.25}
                suffix="%"
                disabled={disabled}
                description="Central bank lending rate"
            />
            <PolicyControl
                label="Minimum CRAR"
                value={policies.minimumCRAR}
                onChange={(v) => onChange("minimumCRAR", v)}
                min={8}
                max={15}
                step={0.5}
                suffix="%"
                disabled={disabled}
                description="Regulatory capital requirement"
            />
            <PolicyControl
                label="Crisis Intervention"
                value={policies.crisisInterventionThreshold}
                onChange={(v) => onChange("crisisInterventionThreshold", v)}
                min={0.3}
                max={0.9}
                step={0.05}
                suffix=""
                disabled={disabled}
                description="System health trigger for intervention"
            />
        </>
    );
}

// Sector Policy Controls
function SectorPolicyControls({
    policies,
    onChange,
    disabled,
}: {
    policies: SectorPolicies;
    onChange: (key: string, value: any) => void;
    disabled: boolean;
}) {
    return (
        <>
            <PolicyControl
                label="Economic Health"
                value={policies.economicHealth}
                onChange={(v) => onChange("economicHealth", v)}
                min={0.1}
                max={1}
                step={0.05}
                suffix=""
                disabled={disabled}
                description="Sector performance level"
            />
            <PolicyControl
                label="Debt Load"
                value={policies.debtLoad}
                onChange={(v) => onChange("debtLoad", v)}
                min={20}
                max={80}
                step={5}
                suffix="%"
                disabled={disabled}
                description="Sector indebtedness"
            />
            <PolicyControl
                label="Volatility"
                value={policies.volatility}
                onChange={(v) => onChange("volatility", v)}
                min={0}
                max={1}
                step={0.05}
                suffix=""
                disabled={disabled}
                description="Performance unpredictability"
            />
        </>
    );
}

// Reusable Policy Control Component
interface PolicyControlProps {
    label: string;
    value: number;
    onChange: (value: number) => void;
    min: number;
    max: number;
    step: number;
    suffix: string;
    disabled: boolean;
    description: string;
}

function PolicyControl({
    label,
    value,
    onChange,
    min,
    max,
    step,
    suffix,
    disabled,
    description,
}: PolicyControlProps) {
    return (
        <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
            <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-semibold text-slate-800">{label}</label>
                <span className="font-mono text-sm font-bold text-blue-600">
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
                disabled={disabled}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <p className="text-xs text-slate-500 mt-1">{description}</p>
        </div>
    );
}
