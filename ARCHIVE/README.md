# ARCHIVE - Legacy & Reference Documentation

This folder contains files that are not required for production but may be useful for reference, testing alternatives, or understanding the development history.

## 📁 Contents

### documentation/
All markdown files documenting development, analysis, and design decisions.

**Files**:
- `ANALYSIS_COMPLETE.md` - Initial analysis results
- `AUDIT_REPORT.md` - Comprehensive system audit
- `DASHBOARD_GUIDE.md` - Dashboard setup guide
- `DETAILED_CHECKLIST.md` - Requirements checklist
- `EXECUTIVE_SUMMARY.md` - High-level summary
- `FIXES_APPLIED.md` - Bug fixes and corrections
- `IMPLEMENTATION_COMPLETE.md` - Implementation status
- `IMPLEMENTATION_SUMMARY.md` - Summary of implementation
- `INDEX.md` - Document index
- `INTEGRATION_README.md` - Integration guide
- `ISSUES_FOUND.md` - Issues and resolutions
- `MODEL_INTEGRATION.md` - Model integration details
- `PHASE_B5_GUIDE.md` - Phase B.5 analysis guide
- `PIPELINE_AUDIT_REPORT.md` - Pipeline audit
- `PIPELINE_ARCHITECTURE.md` - Architecture design
- `PRESENTATION_SUMMARY.md` - Presentation summary
- `PROJECT_COMPLETION.md` - Project completion report
- `REFACTORING_COMPLETE.md` - Refactoring completion
- `REFACTORING_PLAN.md` - Refactoring plan
- `SLA_VIOLATIONS_FIX.md` - SLA violation fixes
- `VERIFICATION_CHECKLIST.md` - Verification checklist
- `docs/` - Additional documentation folder

### legacy_scripts/
Alternative scripts and tools that are not part of the production pipeline.

**Files**:
- `simulate.py` - Legacy simulation script (replaced by run_hybrid_pipeline.py)
- `run_pipeline.py` - Old pipeline orchestrator (replaced by run_hybrid_pipeline.py)
- `verify_integration.py` - Integration verification tool
- `verify_refactoring.py` - Refactoring verification tool
- `analyze_strategy.py` - Strategy analysis tool
- `compare_strategies.py` - Strategy comparison script
- `QUICKSTART.sh` - Quick start shell script
- `scripts/` - Additional scripts folder

---

## 🎯 When to Use These Files

### Documentation
- **Reference**: Understand design decisions and analysis results
- **Troubleshooting**: Find solutions to known issues
- **Learning**: Study the project history and development approach
- **Validation**: Verify requirements and completeness

### Legacy Scripts
- **Testing**: Run alternative autoscaling strategies for comparison
- **Analysis**: Use analyze_strategy.py to compare performance
- **Verification**: Check integration with verify_integration.py
- **Development**: Reference old implementation approaches

---

## 🚀 Production vs Archive

### ✅ Use Production Folder
- `run_hybrid_pipeline.py` - Main pipeline execution
- `autoscaling/hybrid_optimized.py` - Production autoscaler
- `cost/cost_model.py` - Production cost model
- `dashboard/app.py` - Visualization
- `PRODUCTION_README.md` - Getting started guide

### 📦 Use Archive Folder
- Alternative autoscaling strategies (in legacy_scripts/)
- Design documentation (in documentation/)
- Development history and decisions
- Verification and analysis tools

---

## 💡 Key References

### Design Decisions
- `COST_MODEL_SELECTION.md` - Why CloudCostModel selected
- `HYBRID_DEPLOYMENT.md` - HYBRID strategy deployment
- `QUICK_REFERENCE.md` - Common tasks and tips

### Analysis Results
- `PROJECT_COMPLETION.md` - Final completion report
- `PHASE_B5_GUIDE.md` - Phase B.5 analysis results
- `PIPELINE_ARCHITECTURE.md` - System architecture

### Troubleshooting
- `ISSUES_FOUND.md` - Known issues and fixes
- `FIXES_APPLIED.md` - Applied corrections
- `SLA_VIOLATIONS_FIX.md` - SLA violation fixes

---

## 📊 Legacy Strategy Scripts

If you want to test alternative autoscaling strategies (not recommended for production):

```bash
# Copy from archive if needed
cp ARCHIVE/legacy_scripts/simulate.py .
cp ARCHIVE/legacy_scripts/run_pipeline.py .
cp ARCHIVE/legacy_scripts/analyze_strategy.py .

# Run comparison (requires old code)
python ARCHIVE/legacy_scripts/compare_strategies.py
```

---

## 🔄 Organization History

**Before Cleanup**:
```
dataFlow-2026/
├── 21 .md documentation files
├── 7 legacy scripts
├── 50+ various configuration files
└── Multiple __pycache__ directories
```

**After Cleanup** (Current):
```
dataFlow-2026/                  (Production ready)
├── run_hybrid_pipeline.py       ⭐ Essential
├── PRODUCTION_README.md         ⭐ Getting started
├── [core modules]/              ⭐ Core functionality
└── ARCHIVE/                     (Reference only)
    ├── documentation/
    └── legacy_scripts/
```

---

## 📌 Important Notes

1. **Don't delete ARCHIVE**: Keep for reference and historical documentation
2. **Don't move production files**: Keep `run_hybrid_pipeline.py` at root
3. **Use PRODUCTION_README.md**: This is the user guide
4. **Reference documentation**: Use COST_MODEL_SELECTION.md and others as needed

---

## 🚀 To Get Started

See **PRODUCTION_README.md** in the root folder.

```bash
python run_hybrid_pipeline.py --timeframe 15m
```

---

**Archive organized and ready for reference!** 📚
