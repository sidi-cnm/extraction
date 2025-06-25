import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import {
  Users,
  FileText,
  Upload,
  Search,
  Clock,
  TrendingUp,
  Activity,
  Calendar,
  BarChart3,
  ArrowUpRight,
  Plus,
  Eye
} from 'lucide-react';

export default function DashboardPage() {
  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
            <p className="text-gray-600 mt-2">Welcome back! Here's an overview of your practice.</p>
          </div>
          <div className="flex space-x-3">
            <Link href="/patients">
              <Button>
                <Users className="w-4 h-4 mr-2" />
                View Patients
              </Button>
            </Link>
            <Button className="medical-gradient text-white">
              <Plus className="w-4 h-4 mr-2" />
              New Patient
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Patients</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">127</div>
              <p className="text-xs text-muted-foreground">
                <span className="text-green-600 font-medium">+12%</span> from last month
              </p>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Documents</CardTitle>
              <FileText className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">1,429</div>
              <p className="text-xs text-muted-foreground">
                <span className="text-green-600 font-medium">+8%</span> from last month
              </p>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">This Month</CardTitle>
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">89</div>
              <p className="text-xs text-muted-foreground">
                Consultations completed
              </p>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Time Saved</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">24h</div>
              <p className="text-xs text-muted-foreground">
                This month via automation
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Quick Actions */}
        <div className="grid lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
              <CardDescription>
                Frequently used features for efficient workflow
              </CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-4">
              <Link href="/patients">
                <Button variant="outline" className="w-full h-20 flex flex-col items-center justify-center space-y-2 hover:bg-primary/5 hover:border-primary transition-colors">
                  <Users className="w-6 h-6" />
                  <span>View Patients</span>
                </Button>
              </Link>
              
              <Button variant="outline" className="w-full h-20 flex flex-col items-center justify-center space-y-2 hover:bg-primary/5 hover:border-primary transition-colors">
                <Upload className="w-6 h-6" />
                <span>Upload Document</span>
              </Button>
              
              <Button variant="outline" className="w-full h-20 flex flex-col items-center justify-center space-y-2 hover:bg-primary/5 hover:border-primary transition-colors">
                <Search className="w-6 h-6" />
                <span>Search Records</span>
              </Button>
              
              <Button variant="outline" className="w-full h-20 flex flex-col items-center justify-center space-y-2 hover:bg-primary/5 hover:border-primary transition-colors">
                <BarChart3 className="w-6 h-6" />
                <span>View Analytics</span>
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
              <CardDescription>
                Latest updates from your practice
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <div className="flex-1">
                  <p className="text-sm font-medium">New document uploaded for John Smith</p>
                  <p className="text-xs text-gray-500">2 hours ago</p>
                </div>
                <Button variant="ghost" size="sm">
                  <Eye className="w-4 h-4" />
                </Button>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <div className="flex-1">
                  <p className="text-sm font-medium">Emily Johnson's consultation completed</p>
                  <p className="text-xs text-gray-500">4 hours ago</p>
                </div>
                <Button variant="ghost" size="sm">
                  <Eye className="w-4 h-4" />
                </Button>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <div className="flex-1">
                  <p className="text-sm font-medium">Document search performed for Michael Brown</p>
                  <p className="text-xs text-gray-500">1 day ago</p>
                </div>
                <Button variant="ghost" size="sm">
                  <Eye className="w-4 h-4" />
                </Button>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                <div className="flex-1">
                  <p className="text-sm font-medium">New patient registered: Sarah Davis</p>
                  <p className="text-xs text-gray-500">2 days ago</p>
                </div>
                <Button variant="ghost" size="sm">
                  <Eye className="w-4 h-4" />
                </Button>
              </div>

              <div className="pt-2">
                <Button variant="outline" className="w-full">
                  View All Activity
                  <ArrowUpRight className="w-4 h-4 ml-2" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Performance Insights */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5" />
              <span>Performance Insights</span>
            </CardTitle>
            <CardDescription>
              Key metrics and improvements over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center p-6 bg-green-50 rounded-lg">
                <div className="text-3xl font-bold text-green-600 mb-2">60%</div>
                <p className="text-sm text-gray-600">Average time saved per consultation</p>
                <div className="mt-2 flex items-center justify-center text-xs text-green-600">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  <span>+5% this month</span>
                </div>
              </div>
              <div className="text-center p-6 bg-blue-50 rounded-lg">
                <div className="text-3xl font-bold text-blue-600 mb-2">25%</div>
                <p className="text-sm text-gray-600">Increase in daily consultations</p>
                <div className="mt-2 flex items-center justify-center text-xs text-blue-600">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  <span>+3% this month</span>
                </div>
              </div>
              <div className="text-center p-6 bg-purple-50 rounded-lg">
                <div className="text-3xl font-bold text-purple-600 mb-2">98%</div>
                <p className="text-sm text-gray-600">Document processing accuracy</p>
                <div className="mt-2 flex items-center justify-center text-xs text-purple-600">
                  <Activity className="w-3 h-3 mr-1" />
                  <span>Consistent</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recent Patients */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Recent Patients</CardTitle>
              <CardDescription>
                Patients with recent activity or upcoming appointments
              </CardDescription>
            </div>
            <Link href="/patients">
              <Button variant="outline">
                View All
                <ArrowUpRight className="w-4 h-4 ml-2" />
              </Button>
            </Link>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { name: 'John Smith', email: 'john.smith@email.com', lastVisit: '2024-01-15', status: 'Recent Upload' },
                { name: 'Emily Johnson', email: 'emily.johnson@email.com', lastVisit: '2024-01-12', status: 'Consultation' },
                { name: 'Michael Brown', email: 'michael.brown@email.com', lastVisit: '2024-01-10', status: 'Follow-up' },
                { name: 'Sarah Davis', email: 'sarah.davis@email.com', lastVisit: '2024-01-08', status: 'New Patient' }
              ].map((patient, index) => (
                <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-primary/10 rounded-full flex items-center justify-center">
                      <span className="text-sm font-medium text-primary">
                        {patient.name.split(' ').map(n => n[0]).join('')}
                      </span>
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{patient.name}</p>
                      <p className="text-sm text-gray-500">{patient.email}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-900">{patient.status}</p>
                    <p className="text-xs text-gray-500">{patient.lastVisit}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}