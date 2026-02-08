'use client';

import { useQuery } from '@tanstack/react-query';
import { useParams, useRouter } from 'next/navigation';
import { ccpApi } from '@/lib/api/ccp-api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  AlertCircle,
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Network,
  DollarSign,
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

export default function BankDetailPage() {
  const params = useParams();
  const router = useRouter();
  const bankName = decodeURIComponent(params.name as string);

  const { data: bankDetail, isLoading, error } = useQuery({
    queryKey: ['bank-detail', bankName],
    queryFn: () => ccpApi.getBankDetail(bankName),
    enabled: !!bankName,
  });

  const getRiskBadge = (probability: number) => {
    if (probability > 0.7) return { variant: 'destructive' as const, label: 'HIGH RISK' };
    if (probability > 0.5) return { variant: 'default' as const, label: 'ELEVATED RISK' };
    if (probability > 0.3) return { variant: 'secondary' as const, label: 'MEDIUM RISK' };
    return { variant: 'outline' as const, label: 'LOW RISK' };
  };

  if (error) {
    return (
      <div className="space-y-6">
        <Button variant="ghost" onClick={() => router.back()}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to load bank details: {(error as Error).message}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={() => router.back()}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <div>
            <h1 className="text-3xl font-bold">{bankName}</h1>
            <p className="text-muted-foreground">Detailed risk assessment and analytics</p>
          </div>
        </div>
        {bankDetail && (
          <Badge {...getRiskBadge(bankDetail.current_metrics.default_probability)}>
            {getRiskBadge(bankDetail.current_metrics.default_probability).label}
          </Badge>
        )}
      </div>

      {/* Key Metrics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Default Probability</CardTitle>
            <AlertCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : bankDetail ? (
              <>
                <div className="text-2xl font-bold">
                  {(bankDetail.current_metrics.default_probability * 100).toFixed(2)}%
                </div>
                <p className="text-xs text-muted-foreground">ML-predicted risk</p>
              </>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Capital Ratio</CardTitle>
            <TrendingUp className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : bankDetail ? (
              <>
                <div className="text-2xl font-bold">
                  {(bankDetail.current_metrics.capital_ratio * 100).toFixed(2)}%
                </div>
                <p className="text-xs text-muted-foreground">Regulatory capital</p>
              </>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Stress Level</CardTitle>
            <AlertCircle className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : bankDetail ? (
              <>
                <div className="text-2xl font-bold">
                  {(bankDetail.current_metrics.stress_level * 100).toFixed(0)}%
                </div>
                <p className="text-xs text-muted-foreground">Financial stress indicator</p>
              </>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Network Degree</CardTitle>
            <Network className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : bankDetail ? (
              <>
                <div className="text-2xl font-bold">
                  {bankDetail.network_position.degree}
                </div>
                <p className="text-xs text-muted-foreground">
                  Connected to {bankDetail.network_position.neighbors.length} banks
                </p>
              </>
            ) : null}
          </CardContent>
        </Card>
      </div>

      {/* Tabs Content */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="trends">Historical Trends</TabsTrigger>
          <TabsTrigger value="network">Network Position</TabsTrigger>
          <TabsTrigger value="margin">Margin Requirements</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {/* Current Metrics */}
            <Card>
              <CardHeader>
                <CardTitle>Current Financial Metrics</CardTitle>
                <CardDescription>Latest reported values</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {isLoading ? (
                  <>
                    <Skeleton className="h-8" />
                    <Skeleton className="h-8" />
                    <Skeleton className="h-8" />
                  </>
                ) : bankDetail ? (
                  <>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Liquidity Buffer</span>
                      <span className="text-lg font-bold">
                        {(bankDetail.current_metrics.liquidity_buffer * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Leverage Ratio</span>
                      <span className="text-lg font-bold">
                        {bankDetail.current_metrics.leverage.toFixed(2)}x
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Stress Level</span>
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-24 rounded-full bg-muted">
                          <div
                            className="h-2 rounded-full bg-orange-500"
                            style={{
                              width: `${bankDetail.current_metrics.stress_level * 100}%`,
                            }}
                          />
                        </div>
                        <span className="text-sm font-medium">
                          {(bankDetail.current_metrics.stress_level * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </>
                ) : null}
              </CardContent>
            </Card>

            {/* Network Centrality */}
            <Card>
              <CardHeader>
                <CardTitle>Network Centrality Metrics</CardTitle>
                <CardDescription>Systemic importance indicators</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {isLoading ? (
                  <>
                    <Skeleton className="h-8" />
                    <Skeleton className="h-8" />
                    <Skeleton className="h-8" />
                  </>
                ) : bankDetail ? (
                  <>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">PageRank</span>
                      <span className="text-lg font-bold">
                        {bankDetail.network_position.pagerank.toFixed(6)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Betweenness Centrality</span>
                      <span className="text-lg font-bold">
                        {bankDetail.network_position.betweenness.toFixed(6)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Direct Connections</span>
                      <span className="text-lg font-bold">
                        {bankDetail.network_position.degree}
                      </span>
                    </div>
                  </>
                ) : null}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Historical Trends Tab */}
        <TabsContent value="trends" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Capital Ratio Trend</CardTitle>
              <CardDescription>Historical capital adequacy over time</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <Skeleton className="h-[300px]" />
              ) : bankDetail && bankDetail.historical_trend.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={bankDetail.historical_trend}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="capital_ratio"
                      stroke="#10b981"
                      strokeWidth={2}
                      name="Capital Ratio"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                  No historical data available
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Stress Level Trend</CardTitle>
              <CardDescription>Financial stress indicators over time</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <Skeleton className="h-[300px]" />
              ) : bankDetail && bankDetail.historical_trend.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={bankDetail.historical_trend}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="stress_level"
                      stroke="#f59e0b"
                      fill="#f59e0b"
                      fillOpacity={0.3}
                      name="Stress Level"
                    />
                    <Area
                      type="monotone"
                      dataKey="liquidity_buffer"
                      stroke="#3b82f6"
                      fill="#3b82f6"
                      fillOpacity={0.3}
                      name="Liquidity Buffer"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                  No historical data available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Network Position Tab */}
        <TabsContent value="network" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Connected Banks</CardTitle>
              <CardDescription>
                Direct counterparty relationships ({bankDetail?.network_position.neighbors.length || 0} connections)
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-2">
                  {[...Array(5)].map((_, i) => (
                    <Skeleton key={i} className="h-10" />
                  ))}
                </div>
              ) : bankDetail && bankDetail.network_position.neighbors.length > 0 ? (
                <div className="space-y-2 max-h-[400px] overflow-y-auto">
                  {bankDetail.network_position.neighbors.map((neighbor) => (
                    <div
                      key={neighbor}
                      className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 cursor-pointer"
                      onClick={() => router.push(`/ml-dashboard/banks/${encodeURIComponent(neighbor)}`)}
                    >
                      <span className="font-medium">{neighbor}</span>
                      <Button variant="ghost" size="sm">
                        View
                      </Button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-[200px] items-center justify-center text-muted-foreground">
                  No connections found
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Margin Requirements Tab */}
        <TabsContent value="margin" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Margin Requirements</CardTitle>
              <CardDescription>CCP margin and collateral requirements</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-20" />
                  <Skeleton className="h-20" />
                  <Skeleton className="h-20" />
                </div>
              ) : bankDetail?.margin_requirement ? (
                <div className="space-y-6">
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="p-4 border rounded-lg">
                      <p className="text-sm text-muted-foreground">Base Margin</p>
                      <p className="text-2xl font-bold">
                        ${(bankDetail.margin_requirement.base_margin / 1e6).toFixed(2)}M
                      </p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <p className="text-sm text-muted-foreground">Network Add-on</p>
                      <p className="text-2xl font-bold">
                        ${(bankDetail.margin_requirement.network_addon / 1e6).toFixed(2)}M
                      </p>
                    </div>
                    <div className="p-4 border rounded-lg bg-primary/10">
                      <p className="text-sm text-muted-foreground">Total Required</p>
                      <p className="text-2xl font-bold">
                        ${(bankDetail.margin_requirement.total_margin / 1e6).toFixed(2)}M
                      </p>
                    </div>
                  </div>
                  <div className="p-4 bg-muted rounded-lg">
                    <p className="text-sm font-medium mb-2">Explanation</p>
                    <p className="text-sm text-muted-foreground">
                      {bankDetail.margin_requirement.explanation}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="flex h-[200px] items-center justify-center text-muted-foreground">
                  No margin requirement data available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
