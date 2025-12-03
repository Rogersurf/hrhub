import { Controller, Get, Render } from '@nestjs/common';

@Controller()
export class AppController {
  @Get()
  @Render('dashboard')
  getDashboard() {
    return {
      candidateName: 'Candidate #0 - Test'
    };
  }
}