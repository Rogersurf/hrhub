import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MatchingModule } from './matching/matching.module';

@Module({
  imports: [MatchingModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
