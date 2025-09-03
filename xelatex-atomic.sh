#!/bin/bash -e

# Atomic LaTeX compilation script to prevent intermediate file cleanup
# This runs the complete XeLaTeX → BibTeX → XeLaTeX → XeLaTeX sequence atomically

# Debug log file
DEBUG_LOG="/tmp/xelatex-atomic-$(date +%Y%m%d).log"

# Define TeX binaries with absolute paths
XELATEX="/usr/local/texlive/2024/bin/x86_64-linux/xelatex"
BIBTEX="/usr/local/texlive/2024/bin/x86_64-linux/bibtex"

# Check if binaries exist
if [ ! -x "$XELATEX" ]; then
    echo "ERROR: XeLaTeX not found at $XELATEX" | tee -a "$DEBUG_LOG"
    exit 1
fi

if [ ! -x "$BIBTEX" ]; then
    echo "ERROR: BibTeX not found at $BIBTEX" | tee -a "$DEBUG_LOG"
    exit 1
fi

# Print debug information
echo "=== Atomic LaTeX Compilation Started ===" | tee -a "$DEBUG_LOG"
echo "Timestamp: $(date)" | tee -a "$DEBUG_LOG"
echo "Called with args: $@" | tee -a "$DEBUG_LOG"
echo "Working directory: $(pwd)" | tee -a "$DEBUG_LOG"
echo "Process PID: $$" | tee -a "$DEBUG_LOG"
echo "Parent PID: $PPID" | tee -a "$DEBUG_LOG"
echo "XeLaTeX path: $XELATEX" | tee -a "$DEBUG_LOG"
echo "BibTeX path: $BIBTEX" | tee -a "$DEBUG_LOG"

# Extract the main document name
DOC_NAME="$1"
if [[ "$DOC_NAME" == *.tex ]]; then
    DOC_NAME="${DOC_NAME%.tex}"
fi

# Store original working directory and full document path
WORK_DIR="$(pwd)"
DOC_FULL_PATH="${WORK_DIR}/${DOC_NAME}"

echo "Document name: $DOC_NAME" | tee -a "$DEBUG_LOG"
echo "Working directory: $WORK_DIR" | tee -a "$DEBUG_LOG"
echo "Full document path: $DOC_FULL_PATH" | tee -a "$DEBUG_LOG"

# Function to run XeLaTeX
run_xelatex() {
    local step_name="$1"
    echo "=== Running XeLaTeX ($step_name) ===" | tee -a "$DEBUG_LOG"
    
    "$XELATEX" -synctex=1 -interaction=nonstopmode -file-line-error "${DOC_NAME}.tex" 2>&1 | tee -a "$DEBUG_LOG"
    local exit_code=${PIPESTATUS[0]}
    
    echo "XeLaTeX ($step_name) exit code: $exit_code" | tee -a "$DEBUG_LOG"
    
    # Check if log file exists after this step
    if [ -f "${DOC_NAME}.log" ]; then
        local log_size=$(stat -c %s "${DOC_NAME}.log" 2>/dev/null)
        local log_mtime=$(stat -c %y "${DOC_NAME}.log" 2>/dev/null)
        echo "Log file after $step_name: ${DOC_NAME}.log (size: $log_size, modified: $log_mtime)" | tee -a "$DEBUG_LOG"
        tail -n 20 "${DOC_NAME}.log" | tee -a "$DEBUG_LOG"
    else
        echo "WARNING: No log file after $step_name" | tee -a "$DEBUG_LOG"
    fi
    
    # Check if PDF was generated (more important than exit code)
    if [ -f "${DOC_NAME}.pdf" ]; then
        local pdf_size=$(stat -c %s "${DOC_NAME}.pdf" 2>/dev/null)
        local pdf_mtime=$(stat -c %y "${DOC_NAME}.pdf" 2>/dev/null)
        echo "PDF generated in $step_name: ${DOC_NAME}.pdf (size: $pdf_size, modified: $pdf_mtime)" | tee -a "$DEBUG_LOG"
        return 0  # Success if PDF was generated
    else
        echo "ERROR: No PDF generated in $step_name" | tee -a "$DEBUG_LOG"
        return 1  # Failure only if no PDF
    fi
}

# Function to run BibTeX
run_bibtex() {
    echo "=== Running BibTeX ===" | tee -a "$DEBUG_LOG"
    
    if [ -f "${DOC_NAME}.aux" ]; then
        echo "Running BibTeX in directory: $(pwd)" | tee -a "$DEBUG_LOG"
        echo "Document name for BibTeX: $DOC_NAME" | tee -a "$DEBUG_LOG"
        
        "$BIBTEX" "$DOC_NAME" 2>&1 | tee -a "$DEBUG_LOG"
        local exit_code=${PIPESTATUS[0]}
        
        echo "BibTeX exit code: $exit_code" | tee -a "$DEBUG_LOG"
        
        # Check if BBL file was created
        if [ -f "${DOC_NAME}.bbl" ]; then
            local bbl_size=$(stat -c %s "${DOC_NAME}.bbl" 2>/dev/null)
            local bbl_mtime=$(stat -c %y "${DOC_NAME}.bbl" 2>/dev/null)
            echo "BBL file created: ${DOC_NAME}.bbl (size: $bbl_size, modified: $bbl_mtime)" | tee -a "$DEBUG_LOG"
            
            # BibTeX success if BBL was created, regardless of exit code
            return 0
        else
            echo "WARNING: No BBL file created" | tee -a "$DEBUG_LOG"
            return $exit_code
        fi
    else
        echo "ERROR: No AUX file found for BibTeX" | tee -a "$DEBUG_LOG"
        return 1
    fi
}

# Step 1: First XeLaTeX run
echo "Step 1: Initial XeLaTeX compilation" | tee -a "$DEBUG_LOG"
if ! run_xelatex "Step 1"; then
    echo "ERROR: First XeLaTeX run failed" | tee -a "$DEBUG_LOG"
    exit 1
fi

# Step 2: BibTeX run
echo "Step 2: BibTeX processing" | tee -a "$DEBUG_LOG"
if ! run_bibtex; then
    echo "ERROR: BibTeX run failed" | tee -a "$DEBUG_LOG"
    exit 1
fi

# Step 3: Second XeLaTeX run (incorporate bibliography)
echo "Step 3: Second XeLaTeX compilation" | tee -a "$DEBUG_LOG"
if ! run_xelatex "Step 3"; then
    echo "ERROR: Second XeLaTeX run failed" | tee -a "$DEBUG_LOG"
    exit 1
fi

# Step 4: Final XeLaTeX run (resolve cross-references)
echo "Step 4: Final XeLaTeX compilation" | tee -a "$DEBUG_LOG"
if ! run_xelatex "Step 4"; then
    echo "ERROR: Final XeLaTeX run failed" | tee -a "$DEBUG_LOG"
    exit 1
fi

# Final checks
echo "=== Final Status Check ===" | tee -a "$DEBUG_LOG"

if [ -f "${DOC_NAME}.pdf" ]; then
    pdf_size=$(stat -c %s "${DOC_NAME}.pdf" 2>/dev/null)
    pdf_mtime=$(stat -c %y "${DOC_NAME}.pdf" 2>/dev/null)
    echo "PDF generated successfully: ${DOC_NAME}.pdf (size: $pdf_size, modified: $pdf_mtime)" | tee -a "$DEBUG_LOG"
else
    echo "ERROR: No PDF generated" | tee -a "$DEBUG_LOG"
    exit 1
fi

if [ -f "${DOC_NAME}.log" ]; then
    log_size=$(stat -c %s "${DOC_NAME}.log" 2>/dev/null)
    log_mtime=$(stat -c %y "${DOC_NAME}.log" 2>/dev/null)
    echo "Final log file: ${DOC_NAME}.log (size: $log_size, modified: $log_mtime)" | tee -a "$DEBUG_LOG"
else
    echo "WARNING: Final log file missing" | tee -a "$DEBUG_LOG"
fi

if [ -f "${DOC_NAME}.bbl" ]; then
    # Check if citations are resolved
    if grep -q "nimh\|cite\|bibitem" "${DOC_NAME}.bbl"; then
        echo "Citations successfully resolved (bibliography found in BBL)" | tee -a "$DEBUG_LOG"
    else
        echo "WARNING: No citations found in BBL" | tee -a "$DEBUG_LOG"
    fi
else
    echo "WARNING: No BBL file in final state" | tee -a "$DEBUG_LOG"
fi

echo "=== Atomic LaTeX Compilation Completed Successfully ===" | tee -a "$DEBUG_LOG"
echo "" | tee -a "$DEBUG_LOG"

exit 0 