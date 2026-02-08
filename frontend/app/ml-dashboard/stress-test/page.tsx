'use client';

import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { ccpApi } from '@/lib/api/ccp-api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { AlertTriangle, Play, CheckCircle } from 'lucide-react';

export default function StressTestPage() {
  const [config, setConfig] = useState({
    shock_type: 'capital',
    shock_magnitude: 0.2,
    target_banks: [] as string[],
  });

  const { data: banks } = useQuery({
    queryKey: ['ccp-banks'],
    queryFn: () => ccpApi.getAllBanks(),
  });

  const stressTest = useMutation({
    mutationFn: () => ccpApi.runStressTest(config),
  });

  const handleBankToggle = (bankName: string, checked: boolean) => {
    setConfig((prev) => ({
      ...prev,
      target_banks: checked
        ? [...prev.target_banks, bankName]
        : prev.target_banks.filter((b) => b !== bankName),
    }));
  };

  const selectTopRisky = (count: number) => {
    if (!banks) return;
    const sortedBanks = [...banks]
      .sort((a, b) => b.default_probability - a.default_probability)
      .slice(0, count)
      .map((b) => b.bank_name);
    setConfig((prev) => ({ ...prev, target_banks: sortedBanks }));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Stress Testing</h1>
        <p className="text-muted-foreground">
          Apply shocks to test system resilience and contagion effects
        </p>
      </div>

      {/* Status Alerts */}
      {stressTest.isSuccess && (
        <Alert className="border-green-500 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">
            Stress test completed successfully!
          </AlertDescription>
        </Alert>
      )}

      {stressTest.isError && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Stress test failed: {(stressTest.error as Error).message}
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Configuration Panel */}
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Test Configuration</CardTitle>
              <CardDescription>Define the stress scenario parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Shock Type */}
              <div className="space-y-2">
                <Label>Shock Type</Label>
                <Select
                  value={config.shock_type}
                  onValueChange={(value) => setConfig({ ...config, shock_type: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="capital">Capital Shock</SelectItem>
                    <SelectItem value="liquidity">Liquidity Shock</SelectItem>
                    <SelectItem value="market">Market Shock</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {config.shock_type === 'capital' &&
                    'Sudden loss of capital (e.g., loan defaults)'}
                  {config.shock_type === 'liquidity' &&
                    'Liquidity crisis (e.g., funding freeze)'}
                  {config.shock_type === 'market' && 'Market crash (e.g., asset devaluation)'}
                </p>
              </div>

              {/* Shock Magnitude */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Shock Magnitude</Label>
                  <Badge
                    variant={
                      config.shock_magnitude > 0.5
                        ? 'destructive'
                        : config.shock_magnitude > 0.3
                        ? 'default'
                        : 'secondary'
                    }
                  >
                    {(config.shock_magnitude * 100).toFixed(0)}%
                  </Badge>
                </div>
                <Slider
                  value={[config.shock_magnitude]}
                  onValueChange={([value]) => setConfig({ ...config, shock_magnitude: value })}
                  min={0.05}
                  max={1.0}
                  step={0.05}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Mild (5%)</span>
                  <span>Severe (50%)</span>
                  <span>Catastrophic (100%)</span>
                </div>
              </div>

              {/* Quick Selection */}
              <div className="space-y-2">
                <Label>Quick Selection</Label>
                <div className="flex gap-2 flex-wrap">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => selectTopRisky(1)}
                  >
                    Top 1 Risky
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => selectTopRisky(3)}
                  >
                    Top 3 Risky
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => selectTopRisky(5)}
                  >
                    Top 5 Risky
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setConfig((prev) => ({ ...prev, target_banks: [] }))}
                  >
                    Clear All
                  </Button>
                </div>
              </div>

              {/* Run Test Button */}
              <Button
                onClick={() => stressTest.mutate()}
                disabled={stressTest.isPending || config.target_banks.length === 0}
                className="w-full"
                size="lg"
              >
                {stressTest.isPending ? (
                  <>Running Test...</>
                ) : (
                  <>
                    <Play className="mr-2 h-5 w-5" />
                    Run Stress Test
                  </>
                )}
              </Button>

              {config.target_banks.length === 0 && (
                <p className="text-sm text-muted-foreground text-center">
                  Select at least one bank to run the stress test
                </p>
              )}
            </CardContent>
          </Card>

          {/* Bank Selection */}
          <Card>
            <CardHeader>
              <CardTitle>Target Banks</CardTitle>
              <CardDescription>
                Select banks to apply the shock ({config.target_banks.length} selected)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="max-h-[400px] overflow-y-auto space-y-2">
                {banks?.map((bank) => (
                  <div
                    key={bank.bank_name}
                    className="flex items-center space-x-2 p-2 border rounded-lg hover:bg-muted/50"
                  >
                    <Checkbox
                      id={bank.bank_name}
                      checked={config.target_banks.includes(bank.bank_name)}
                      onCheckedChange={(checked) =>
                        handleBankToggle(bank.bank_name, checked as boolean)
                      }
                    />
                    <label
                      htmlFor={bank.bank_name}
                      className="flex-1 cursor-pointer text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      {bank.bank_name}
                    </label>
                    <Badge
                      variant={
                        bank.default_probability > 0.7
                          ? 'destructive'
                          : bank.default_probability > 0.3
                          ? 'default'
                          : 'secondary'
                      }
                      className="text-xs"
                    >
                      {(bank.default_probability * 100).toFixed(1)}%
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Info Panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>About Stress Testing</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">Purpose</h4>
                <p className="text-sm text-muted-foreground">
                  Stress testing evaluates how shocks to specific banks propagate through the
                  network, helping identify:
                </p>
                <ul className="list-disc list-inside text-sm text-muted-foreground mt-2 space-y-1">
                  <li>Cascade potential</li>
                  <li>Vulnerable institutions</li>
                  <li>Network breaking points</li>
                  <li>Systemic risk amplification</li>
                </ul>
              </div>

              <div>
                <h4 className="font-medium mb-2">Shock Types</h4>
                <div className="space-y-2">
                  <div>
                    <p className="text-sm font-medium">Capital Shock</p>
                    <p className="text-xs text-muted-foreground">
                      Reduces bank capital ratios, increasing default risk
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Liquidity Shock</p>
                    <p className="text-xs text-muted-foreground">
                      Drains liquidity buffers, forcing asset sales
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Market Shock</p>
                    <p className="text-xs text-muted-foreground">
                      Market-wide crash affecting all exposures
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Regulatory Context</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Stress testing is required by Basel III for systemic risk assessment. CCPs must
                demonstrate ability to withstand:
              </p>
              <ul className="list-disc list-inside text-sm text-muted-foreground mt-2 space-y-1">
                <li>Default of largest counterparty</li>
                <li>Default of top-2 counterparties</li>
                <li>Extreme but plausible scenarios</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Results Section */}
      {stressTest.isSuccess && stressTest.data && (
        <Card>
          <CardHeader>
            <CardTitle>Test Results</CardTitle>
            <CardDescription>Impact analysis of the stress scenario</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid gap-4 md:grid-cols-3">
                <div className="p-4 border rounded-lg">
                  <p className="text-sm text-muted-foreground">Banks Affected</p>
                  <p className="text-2xl font-bold">
                    {stressTest.data.results?.banks_affected || 0}
                  </p>
                </div>
                <div className="p-4 border rounded-lg">
                  <p className="text-sm text-muted-foreground">Cascade Size</p>
                  <p className="text-2xl font-bold">
                    {stressTest.data.results?.cascade_size || 0}
                  </p>
                </div>
                <div className="p-4 border rounded-lg">
                  <p className="text-sm text-muted-foreground">Status</p>
                  <p className="text-xl font-bold text-green-600">
                    {stressTest.data.status || 'Completed'}
                  </p>
                </div>
              </div>
              {stressTest.data.results?.message && (
                <Alert>
                  <AlertDescription>{stressTest.data.results.message}</AlertDescription>
                </Alert>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
