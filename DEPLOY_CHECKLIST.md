# Deployment Checklist for FinSureAI

Use this checklist to ensure a smooth deployment to Vercel.

## Pre-Deployment ✅

- [ ] All code is committed to git
- [ ] `.env.example` file exists with all required environment variables
- [ ] `.gitignore` is properly configured
- [ ] `package.json` has correct build scripts
- [ ] Code is pushed to GitHub

## Supabase Setup ✅

- [ ] Supabase project created
- [ ] `jobs` table created with correct schema
- [ ] Storage bucket `datasets` created with public access
- [ ] Supabase URL noted down
- [ ] Anon key noted down
- [ ] Service role key noted down (keep secret!)

## Modal Setup ✅

- [ ] Modal account created
- [ ] Modal CLI installed (`pip install modal`)
- [ ] Modal authenticated (`modal token new`)
- [ ] `modal/finetune_job.py` deployed
- [ ] `modal/inference_job.py` deployed
- [ ] Modal API credentials noted down

## Vercel Deployment ✅

- [ ] Vercel account created
- [ ] GitHub repository connected to Vercel
- [ ] Framework preset set to Next.js
- [ ] Environment variables added:
  - [ ] `NEXT_PUBLIC_SUPABASE_URL`
  - [ ] `NEXT_PUBLIC_SUPABASE_ANON_KEY`
  - [ ] `SUPABASE_SERVICE_ROLE_KEY`
  - [ ] `SUPABASE_BUCKET`
  - [ ] `MODAL_API_TOKEN`
  - [ ] `MODAL_API_SECRET`
- [ ] Deployment successful
- [ ] No build errors

## Post-Deployment Testing ✅

- [ ] Application loads successfully
- [ ] User registration works
- [ ] User login works
- [ ] Dashboard is accessible
- [ ] File upload works
- [ ] Model selection works
- [ ] Finetuning job triggers successfully
- [ ] Job status updates correctly
- [ ] Chat interface loads
- [ ] Inference works with fine-tuned model
- [ ] Sign out works

## Optional Enhancements 🚀

- [ ] Custom domain configured
- [ ] Analytics added (e.g., Vercel Analytics, PostHog)
- [ ] Error monitoring added (e.g., Sentry)
- [ ] Performance monitoring enabled
- [ ] Database backups configured
- [ ] Rate limiting implemented
- [ ] API documentation added

## Troubleshooting Guide

### Build Fails
1. Check all environment variables are set
2. Verify `package.json` dependencies are correct
3. Check build logs in Vercel dashboard

### Database Connection Issues
1. Verify Supabase URL and keys
2. Check if table schema matches expectations
3. Ensure service role key has correct permissions

### Modal Job Fails
1. Check Modal deployment status
2. Verify Modal API credentials
3. Check Modal function logs

### Storage Upload Fails
1. Verify storage bucket exists
2. Check bucket permissions (should be public)
3. Verify bucket name matches `SUPABASE_BUCKET` env var

## Maintenance

- Monitor Vercel deployment status
- Check Modal function usage and costs
- Review Supabase storage usage
- Update dependencies regularly
- Monitor application logs for errors

