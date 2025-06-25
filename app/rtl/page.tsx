import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Languages, Globe, ArrowRight, ArrowLeft, CheckCircle } from 'lucide-react';

export default function RTLPage() {
  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">RTL Support</h1>
            <p className="text-gray-600 mt-2">Right-to-left language support and internationalization</p>
          </div>
          <Badge variant="secondary" className="bg-blue-100 text-blue-800">
            Coming Soon
          </Badge>
        </div>

        {/* Language Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Languages className="w-5 h-5" />
              <span>Language Settings</span>
            </CardTitle>
            <CardDescription>
              Configure language preferences and text direction for the application
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              {/* Current Language */}
              <div className="space-y-4">
                <h3 className="font-medium text-gray-900">Current Language</h3>
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                      <span className="text-sm font-medium text-blue-600">EN</span>
                    </div>
                    <div>
                      <p className="font-medium">English</p>
                      <p className="text-sm text-gray-500">Left-to-right</p>
                    </div>
                  </div>
                  <CheckCircle className="w-5 h-5 text-green-600" />
                </div>
              </div>

              {/* Available Languages */}
              <div className="space-y-4">
                <h3 className="font-medium text-gray-900">Available Languages</h3>
                <div className="space-y-2">
                  {[
                    { code: 'AR', name: 'Arabic', direction: 'Right-to-left', available: false },
                    { code: 'HE', name: 'Hebrew', direction: 'Right-to-left', available: false },
                    { code: 'FA', name: 'Persian', direction: 'Right-to-left', available: false },
                    { code: 'FR', name: 'French', direction: 'Left-to-right', available: false },
                  ].map((lang) => (
                    <div key={lang.code} className="flex items-center justify-between p-3 border rounded-lg opacity-50">
                      <div className="flex items-center space-x-3">
                        <div className="w-6 h-6 bg-gray-100 rounded-full flex items-center justify-center">
                          <span className="text-xs font-medium text-gray-600">{lang.code}</span>
                        </div>
                        <div>
                          <p className="font-medium text-gray-700">{lang.name}</p>
                          <p className="text-xs text-gray-500">{lang.direction}</p>
                        </div>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        Coming Soon
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* RTL Preview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Globe className="w-5 h-5" />
              <span>RTL Layout Preview</span>
            </CardTitle>
            <CardDescription>
              Preview how the interface will look with right-to-left text direction
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="bg-gray-50 p-6 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-medium text-gray-900">RTL Preview Mode</h4>
                <Switch disabled />
              </div>
              
              {/* Mock RTL Interface */}
              <div className="bg-white p-4 rounded-lg border-2 border-dashed border-gray-300">
                <div className="flex items-center justify-between mb-4" dir="rtl">
                  <div className="flex items-center space-x-2 space-x-reverse">
                    <ArrowLeft className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-600">عودة إلى المرضى</span>
                  </div>
                  <Button size="sm" disabled>
                    <span className="ml-2">رفع مستند</span>
                    <ArrowRight className="w-4 h-4" />
                  </Button>
                </div>
                
                <div className="space-y-3" dir="rtl">
                  <div className="flex items-center justify-between">
                    <h5 className="font-medium">معلومات المريض</h5>
                    <Badge className="bg-green-100 text-green-800">نشط</Badge>
                  </div>
                  <div className="text-sm text-gray-600">
                    <p>الاسم: أحمد محمد علي</p>
                    <p>البريد الإلكتروني: ahmed@example.com</p>
                  </div>
                </div>
              </div>
              
              <p className="text-sm text-gray-500 mt-4">
                This is a preview of how the interface would appear with Arabic text and RTL layout direction.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Implementation Status */}
        <Card>
          <CardHeader>
            <CardTitle>Implementation Roadmap</CardTitle>
            <CardDescription>
              Planned features for comprehensive internationalization support
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { 
                  feature: 'Text Direction Support',
                  description: 'Automatic RTL/LTR layout switching based on language',
                  status: 'planned'
                },
                { 
                  feature: 'Arabic Language Pack',
                  description: 'Complete Arabic translation for healthcare terminology',
                  status: 'planned'
                },
                { 
                  feature: 'Hebrew Language Pack',
                  description: 'Hebrew translation with medical terminology support',
                  status: 'planned'
                },
                { 
                  feature: 'Date & Number Formatting',
                  description: 'Locale-aware formatting for dates, numbers, and currencies',
                  status: 'planned'
                },
                { 
                  feature: 'Medical Term Translation',
                  description: 'Specialized medical vocabulary translation database',
                  status: 'planned'
                },
                { 
                  feature: 'Document OCR RTL Support',
                  description: 'Right-to-left text recognition in uploaded documents',
                  status: 'research'
                }
              ].map((item, index) => (
                <div key={index} className="flex items-start space-x-4 p-4 bg-gray-50 rounded-lg">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                  <div className="flex-1">
                    <h5 className="font-medium text-gray-900">{item.feature}</h5>
                    <p className="text-sm text-gray-600 mt-1">{item.description}</p>
                  </div>
                  <Badge 
                    variant="outline"
                    className={`text-xs ${
                      item.status === 'planned' 
                        ? 'bg-blue-50 text-blue-700 border-blue-200' 
                        : 'bg-yellow-50 text-yellow-700 border-yellow-200'
                    }`}
                  >
                    {item.status === 'planned' ? 'Planned' : 'Research'}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Contact for RTL Support */}
        <Card>
          <CardHeader>
            <CardTitle>Need RTL Support?</CardTitle>
            <CardDescription>
              Contact our team to discuss priority implementation for your region
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button className="medical-gradient text-white">
                Request RTL Implementation
              </Button>
              <Button variant="outline">
                View Documentation
              </Button>
            </div>
            <p className="text-sm text-gray-500 mt-4">
              RTL support is being developed to serve healthcare professionals in Arabic, Hebrew, 
              and other right-to-left language regions. Priority implementation available for enterprise customers.
            </p>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}