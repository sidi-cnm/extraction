// Landing page component with comprehensive content and branding
// Showcases SanaMed's value proposition and benefits

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { AuthGuard } from '@/components/auth/auth-guard';
import {
  Stethoscope,
  Clock,
  Shield,
  Zap,
  FileText,
  Search,
  Upload,
  BarChart3,
  Mail,
  Phone,
  MapPin,
  DollarSign,
  TrendingUp,
  Users,
  CheckCircle,
  QrCode,
  Facebook,
  Chrome,
  Apple
} from 'lucide-react';

export default function HomePage() {
  return (
    <AuthGuard requireAuth={false}>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-teal-50">
        {/* Header */}
        <header className="container mx-auto px-4 py-6">
          <nav className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 medical-gradient rounded-lg flex items-center justify-center">
                <Stethoscope className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-bold text-gray-900">SANAMED</span>
            </div>
            <div className="space-x-4">
              <Link href="/auth/login">
                <Button variant="ghost">Sign In</Button>
              </Link>
              <Link href="/auth/signup">
                <Button className="medical-gradient text-white">Get Started</Button>
              </Link>
            </div>
          </nav>
        </header>

        {/* Hero Section */}
        <section className="container mx-auto px-4 py-20 text-center">
          <div className="max-w-5xl mx-auto">
            <h1 className="text-6xl font-bold text-gray-900 mb-6">
              <span className="text-primary">Focus on Patients,</span><br />
              We Process the Data
            </h1>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed max-w-3xl mx-auto">
              SANAMED simplifies medical record management for healthcare professionals.
              Our AI-powered platform provides instant access to patient histories, intelligent
              document analysis, and seamless workflow integration - reducing admin time and
              optimizing consultations.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
              <Link href="/auth/signup">
                <Button size="lg" className="medical-gradient text-white px-8 py-4 text-lg">
                  Start Free Trial
                </Button>
              </Link>
              <Link href="/auth/login">
                <Button size="lg" variant="outline" className="px-8 py-4 text-lg">
                  Demo Login
                </Button>
              </Link>
            </div>

            {/* Value Proposition Stats */}
            <div className="grid md:grid-cols-4 gap-6 bg-white rounded-2xl p-8 shadow-lg">
              <div className="text-center">
                <div className="text-3xl font-bold text-primary mb-2">+45 min/day</div>
                <p className="text-sm text-gray-600">Freed from admin tasks</p>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-primary mb-2">4-8 more</div>
                <p className="text-sm text-gray-600">Consultations per day</p>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-primary mb-2">+$2,050</div>
                <p className="text-sm text-gray-600">Additional revenue/month</p>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-primary mb-2">12-18 months</div>
                <p className="text-sm text-gray-600">Fast ROI timeline</p>
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="container mx-auto px-4 py-20">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              You cure, <span className="text-primary">we analyse</span>
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our comprehensive platform addresses the key challenges healthcare professionals
              face in document management and patient data access.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="text-center hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 medical-gradient rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Clock className="w-6 h-6 text-white" />
                </div>
                <CardTitle>Save Time</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Reduce administrative tasks by 60% with automated document processing
                  and intelligent search capabilities.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="text-center hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 medical-gradient rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Search className="w-6 h-6 text-white" />
                </div>
                <CardTitle>Instant Search</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Find relevant patient information instantly with AI-powered semantic
                  search across all medical documents.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="text-center hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 medical-gradient rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Shield className="w-6 h-6 text-white" />
                </div>
                <CardTitle>Secure & Compliant</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  HIPAA-compliant platform with enterprise-grade security and
                  data encryption for patient confidentiality.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="text-center hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 medical-gradient rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Zap className="w-6 h-6 text-white" />
                </div>
                <CardTitle>Fast ROI</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Increase consultation capacity by 25% and see return on investment
                  within the first 12-18 months of implementation.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Key Features */}
        <section className="bg-white py-20">
          <div className="container mx-auto px-4">
            <div className="grid lg:grid-cols-2 gap-12 items-center">
              <div>
                <h2 className="text-4xl font-bold text-gray-900 mb-6">
                  Intelligent Document Management
                </h2>
                <div className="space-y-6">
                  <div className="flex items-start space-x-4">
                    <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Upload className="w-4 h-4 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-2">Smart Upload & Processing</h3>
                      <p className="text-gray-600">
                        Automatically extract key information from medical documents using advanced NLP technology.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-4">
                    <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                      <FileText className="w-4 h-4 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-2">Comprehensive Summaries</h3>
                      <p className="text-gray-600">
                        Get AI-generated summaries of patient documents for quick review and decision-making.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-4">
                    <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                      <BarChart3 className="w-4 h-4 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-2">Analytics & Insights</h3>
                      <p className="text-gray-600">
                        Track patient trends and optimize your practice with detailed analytics and reporting.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-primary/5 to-primary/10 p-8 rounded-2xl medical-pattern">
                <div className="aspect-video bg-white rounded-lg shadow-lg flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-16 h-16 medical-gradient rounded-full flex items-center justify-center mx-auto mb-4">
                      <Stethoscope className="w-8 h-8 text-white" />
                    </div>
                    <p className="text-gray-600 font-medium">Interactive Demo</p>
                    <p className="text-sm text-gray-500">Experience SanaMed in action</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ROI Section */}
        <section className="container mx-auto px-4 py-20">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Investment & Returns
            </h2>
            <p className="text-lg text-gray-600">
              See how SanaMed pays for itself and generates additional revenue
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 mb-12">
            <Card className="text-center">
              <CardHeader>
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <DollarSign className="w-6 h-6 text-green-600" />
                </div>
                <CardTitle className="text-green-600">Monthly Investment</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-green-600 mb-2">$256</div>
                <p className="text-gray-600">SanaMed subscription</p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardHeader>
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <TrendingUp className="w-6 h-6 text-blue-600" />
                </div>
                <CardTitle className="text-blue-600">Monthly Returns</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-blue-600 mb-2">$2,050</div>
                <p className="text-gray-600">Additional revenue</p>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardHeader>
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Users className="w-6 h-6 text-purple-600" />
                </div>
                <CardTitle className="text-purple-600">Net Benefit</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-purple-600 mb-2">$1,794</div>
                <p className="text-gray-600">Monthly profit increase</p>
              </CardContent>
            </Card>
          </div>

          <div className="bg-primary/5 p-8 rounded-2xl text-center">
            <CheckCircle className="w-12 h-12 text-primary mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 mb-2">
              ROI Achievement: 12-18 Months
            </h3>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Start seeing returns immediately with increased efficiency, and achieve full ROI 
              within 12-18 months through enhanced consultation capacity and reduced overhead.
            </p>
          </div>
        </section>

        {/* Contact Section */}
        <section className="bg-gray-900 text-white py-20">
          <div className="container mx-auto px-4">
            <div className="grid lg:grid-cols-2 gap-12">
              <div>
                <h2 className="text-4xl font-bold mb-6">CONTACT US</h2>
                <p className="text-gray-300 mb-8 text-lg">
                  Ready to transform your medical practice? Get in touch with our team
                  for a personalized demonstration and implementation plan.
                </p>
                
                <div className="space-y-4 mb-8">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
                      <Phone className="w-5 h-5 text-primary" />
                    </div>
                    <span className="text-lg">+123-456-7890</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
                      <Mail className="w-5 h-5 text-primary" />
                    </div>
                    <span className="text-lg">contact.sanamed@gmail.com</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
                      <MapPin className="w-5 h-5 text-primary" />
                    </div>
                    <span className="text-lg">Milan, Italy</span>
                  </div>
                </div>

                {/* QR Code placeholder */}
                <div className="bg-white p-4 rounded-lg w-32 h-32 flex items-center justify-center">
                  <QrCode className="w-20 h-20 text-gray-400" />
                </div>
              </div>
              
              <div>
                <Card>
                  <CardHeader>
                    <CardTitle>Request a Demo</CardTitle>
                    <CardDescription>
                      See SanaMed in action with a personalized demonstration
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <input
                      type="text"
                      placeholder="Your Name"
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-gray-900"
                    />
                    <input
                      type="email"
                      placeholder="Email Address"
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-gray-900"
                    />
                    <input
                      type="text"
                      placeholder="Practice Name"
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-gray-900"
                    />
                    <input
                      type="tel"
                      placeholder="Phone Number"
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-gray-900"
                    />
                    <Button className="w-full medical-gradient text-white py-3 text-lg">
                      Schedule Demo
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 py-12">
          <div className="container mx-auto px-4">
            <div className="grid md:grid-cols-4 gap-8">
              <div>
                <div className="flex items-center space-x-2 mb-4">
                  <div className="w-8 h-8 medical-gradient rounded-lg flex items-center justify-center">
                    <Stethoscope className="w-5 h-5 text-white" />
                  </div>
                  <span className="text-xl font-bold text-gray-900">SANAMED</span>
                </div>
                <p className="text-gray-600 text-sm">
                  Revolutionizing medical record management for healthcare professionals worldwide.
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold text-gray-900 mb-4">Product</h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li><a href="#" className="hover:text-primary">Features</a></li>
                  <li><a href="#" className="hover:text-primary">Pricing</a></li>
                  <li><a href="#" className="hover:text-primary">Security</a></li>
                  <li><a href="#" className="hover:text-primary">Integrations</a></li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold text-gray-900 mb-4">Support</h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li><a href="#" className="hover:text-primary">Documentation</a></li>
                  <li><a href="#" className="hover:text-primary">Help Center</a></li>
                  <li><a href="#" className="hover:text-primary">Training</a></li>
                  <li><a href="#" className="hover:text-primary">Contact</a></li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold text-gray-900 mb-4">Company</h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li><a href="#" className="hover:text-primary">About</a></li>
                  <li><a href="#" className="hover:text-primary">Careers</a></li>
                  <li><a href="#" className="hover:text-primary">Privacy</a></li>
                  <li><a href="#" className="hover:text-primary">Terms</a></li>
                </ul>
              </div>
            </div>
            
            <div className="border-t border-gray-200 mt-8 pt-8 flex items-center justify-between">
              <p className="text-gray-600 text-sm">
                © 2024 SanaMed. All rights reserved.
              </p>
              <div className="flex space-x-4">
                <span className="text-sm text-gray-500">Made with ❤️ for healthcare professionals</span>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </AuthGuard>
  );
}