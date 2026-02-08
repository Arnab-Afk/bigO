'use client';

import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { ccpApi, BankRiskScore } from '@/lib/api/ccp-api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle, Download, Search, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';

type SortField = keyof BankRiskScore;
type SortDirection = 'asc' | 'desc';

export default function BanksPage() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState('');
  const [riskFilter, setRiskFilter] = useState<string>('all');
  const [sortField, setSortField] = useState<SortField>('default_probability');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const { data: banks, isLoading, error } = useQuery({
    queryKey: ['ccp-banks'],
    queryFn: () => ccpApi.getAllBanks(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Filter and sort banks
  const filteredAndSortedBanks = useMemo(() => {
    if (!banks) return [];

    let filtered = banks.filter((bank) => {
      const matchesSearch = bank.bank_name
        .toLowerCase()
        .includes(searchQuery.toLowerCase());
      const matchesRisk =
        riskFilter === 'all' ||
        bank.risk_tier.toLowerCase().includes(riskFilter.toLowerCase());
      return matchesSearch && matchesRisk;
    });

    filtered.sort((a, b) => {
      const aValue = a[sortField];
      const bValue = b[sortField];
      const direction = sortDirection === 'asc' ? 1 : -1;

      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return (aValue - bValue) * direction;
      }
      return String(aValue).localeCompare(String(bValue)) * direction;
    });

    return filtered;
  }, [banks, searchQuery, riskFilter, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <ArrowUpDown className="ml-2 h-4 w-4 opacity-50" />;
    return sortDirection === 'asc' ? (
      <ArrowUp className="ml-2 h-4 w-4" />
    ) : (
      <ArrowDown className="ml-2 h-4 w-4" />
    );
  };

  const getRiskBadgeVariant = (tier: string) => {
    if (tier.includes('1')) return 'destructive';
    if (tier.includes('2')) return 'default';
    if (tier.includes('3')) return 'secondary';
    return 'outline';
  };

  const exportToCSV = () => {
    if (!filteredAndSortedBanks.length) return;

    const headers = [
      'Bank Name',
      'Default Probability',
      'Risk Tier',
      'Capital Ratio',
      'Stress Level',
      'PageRank',
    ];
    const csvContent = [
      headers.join(','),
      ...filteredAndSortedBanks.map((bank) =>
        [
          bank.bank_name,
          bank.default_probability.toFixed(4),
          bank.risk_tier,
          bank.capital_ratio.toFixed(4),
          bank.stress_level.toFixed(4),
          bank.pagerank.toFixed(6),
        ].join(',')
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `banks-risk-analysis-${new Date().toISOString()}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Bank Risk Analysis</h1>
        <p className="text-muted-foreground">
          Comprehensive risk assessment for all banking institutions
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to load banks: {(error as Error).message}
          </AlertDescription>
        </Alert>
      )}

      {/* Filters and Search */}
      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
          <CardDescription>Search and filter banking institutions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4 md:flex-row">
            {/* Search */}
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search banks..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-8"
                />
              </div>
            </div>

            {/* Risk Filter */}
            <Select value={riskFilter} onValueChange={setRiskFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Filter by risk" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Risk Levels</SelectItem>
                <SelectItem value="tier 1">Tier 1 (Highest)</SelectItem>
                <SelectItem value="tier 2">Tier 2 (High)</SelectItem>
                <SelectItem value="tier 3">Tier 3 (Medium)</SelectItem>
                <SelectItem value="tier 4">Tier 4 (Low)</SelectItem>
              </SelectContent>
            </Select>

            {/* Export Button */}
            <Button
              onClick={exportToCSV}
              variant="outline"
              disabled={!filteredAndSortedBanks.length}
            >
              <Download className="mr-2 h-4 w-4" />
              Export CSV
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results Count */}
      {!isLoading && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Showing {filteredAndSortedBanks.length} of {banks?.length || 0} banks
          </p>
        </div>
      )}

      {/* Banks Table */}
      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>
                  <Button
                    variant="ghost"
                    onClick={() => handleSort('bank_name')}
                    className="h-auto p-0 hover:bg-transparent"
                  >
                    Bank Name
                    <SortIcon field="bank_name" />
                  </Button>
                </TableHead>
                <TableHead>
                  <Button
                    variant="ghost"
                    onClick={() => handleSort('default_probability')}
                    className="h-auto p-0 hover:bg-transparent"
                  >
                    Default Probability
                    <SortIcon field="default_probability" />
                  </Button>
                </TableHead>
                <TableHead>Risk Tier</TableHead>
                <TableHead>
                  <Button
                    variant="ghost"
                    onClick={() => handleSort('capital_ratio')}
                    className="h-auto p-0 hover:bg-transparent"
                  >
                    Capital Ratio
                    <SortIcon field="capital_ratio" />
                  </Button>
                </TableHead>
                <TableHead>
                  <Button
                    variant="ghost"
                    onClick={() => handleSort('stress_level')}
                    className="h-auto p-0 hover:bg-transparent"
                  >
                    Stress Level
                    <SortIcon field="stress_level" />
                  </Button>
                </TableHead>
                <TableHead>
                  <Button
                    variant="ghost"
                    onClick={() => handleSort('pagerank')}
                    className="h-auto p-0 hover:bg-transparent"
                  >
                    PageRank
                    <SortIcon field="pagerank" />
                  </Button>
                </TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                // Loading skeletons
                <>
                  {[...Array(10)].map((_, i) => (
                    <TableRow key={i}>
                      <TableCell><Skeleton className="h-4 w-[200px]" /></TableCell>
                      <TableCell><Skeleton className="h-4 w-[80px]" /></TableCell>
                      <TableCell><Skeleton className="h-5 w-[60px]" /></TableCell>
                      <TableCell><Skeleton className="h-4 w-[80px]" /></TableCell>
                      <TableCell><Skeleton className="h-4 w-[80px]" /></TableCell>
                      <TableCell><Skeleton className="h-4 w-[80px]" /></TableCell>
                      <TableCell><Skeleton className="h-8 w-[80px]" /></TableCell>
                    </TableRow>
                  ))}
                </>
              ) : filteredAndSortedBanks.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center text-muted-foreground">
                    No banks found matching your criteria
                  </TableCell>
                </TableRow>
              ) : (
                // Data rows
                filteredAndSortedBanks.map((bank) => (
                  <TableRow
                    key={bank.bank_name}
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => router.push(`/ml-dashboard/banks/${encodeURIComponent(bank.bank_name)}`)}
                  >
                    <TableCell className="font-medium">{bank.bank_name}</TableCell>
                    <TableCell>
                      <span
                        className={
                          bank.default_probability > 0.7
                            ? 'text-red-600 font-semibold'
                            : bank.default_probability > 0.3
                            ? 'text-orange-600 font-semibold'
                            : 'text-green-600 font-semibold'
                        }
                      >
                        {(bank.default_probability * 100).toFixed(2)}%
                      </span>
                    </TableCell>
                    <TableCell>
                      <Badge variant={getRiskBadgeVariant(bank.risk_tier)}>
                        {bank.risk_tier}
                      </Badge>
                    </TableCell>
                    <TableCell>{(bank.capital_ratio * 100).toFixed(2)}%</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-20 rounded-full bg-muted">
                          <div
                            className="h-2 rounded-full bg-primary"
                            style={{ width: `${bank.stress_level * 100}%` }}
                          />
                        </div>
                        {(bank.stress_level * 100).toFixed(0)}%
                      </div>
                    </TableCell>
                    <TableCell>{bank.pagerank.toFixed(6)}</TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          router.push(`/ml-dashboard/banks/${encodeURIComponent(bank.bank_name)}`);
                        }}
                      >
                        View Details
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
