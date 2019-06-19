import numpy as np
import json
import os, copy, re
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


def jupyter_replace_tags(fname_tmpl, replace, outfile=None, overwrite=False, 
                         verbose=False):
    """
    Search through a Jupyter notebook file for tagged lines, replace them with 
    new values, and then save into a new Jupyter notebook.
    
    Tags work in a simple way: If any line in the notebook has an in-line 
    comment of the form '# @tag', and a key 'tag' exists in the 'replace' dict, 
    the entire line will be replaced with 'tag = value', where 'value' is the 
    value of 'replace[key]', which is assumed to be a string. Any expression 
    that was on that line will be completely replaced.
    
    Parameters
    ----------
    fname_tmpl : str
        Filename of input notebook that contains tags to be replaced.
    
    replace : dict
        Dictionary of tag:value pairs. The values will all be inserted as 
        Python strings, and so the code in the Jupyter notebook should be 
        prepared to do a type conversion if necessary.
    
    outfile : str, optional
        Filename to save the tag-replaced notebook to. If not specified, the 
        updated JSON dict will be returned from this function. Default: None.
    
    overwrite : bool, optional
        If outfile is not None, whether to overwrite a notebook file if one 
        with the same filename already exists. Default: False.
    
    verbose : bool, optional
        If True, print out tags as they are found. Default: False.
    
    Returns
    -------
    new_tree : JSON dict, optional
        If outfile=None, a dict containing the updated JSON data for the 
        notebook is returned.
    """
    # Load Jupyter notebook as JSON file
    with open(fname_tmpl, 'r') as f:
        tree = json.load(f)
    new_tree = copy.copy(tree)
    if verbose:
        print("jupyter_replace_tags(): Running on '{}'".format(fname_tmpl))
    
    # Loop over cells and replace tagged strings
    num_cells = 0; replaced = 0
    for i, cell in enumerate(new_tree['cells']):
        num_cells += 1
        
        # Loop over lines in cell
        for j, line in enumerate(cell['source']):
        
            # Check for tag, denoted by an '@'
            if '@' in line:
                
                # Parse tag using regex
                p = re.compile("@\w+")
                tags = p.search(line)
                if tags is None: continue # ignore floating '@' symbols
                
                # Get tag name (only keep first if >1 found)
                tag = tags.group(0)[1:]
            
                # Check if tag exists in replace dict and then do replacement
                if tag in replace.keys():
                    
                    # Do replacement
                    if verbose: print("  Found valid tag:", tag)
                    replaced += 1
                    new_tree['cells'][i]['source'][j] \
                        = "{} = \"{}\"\n".format(tag, replace[tag])
                else:
                    if verbose: print("  Found unmatched tag:", tag)
                    
    # Report status
    if verbose:
        print("  Number of cells: %d" % num_cells)
        print("  Replacements made: %d" % replaced)
    
    # Either save or return notebook data
    if outfile is not None:
        if os.path.exists(outfile) and not overwrite:
            raise OSError(
                "File '{}' already exists and overwrite=False.".format(outfile))
        with open(outfile, 'w') as f:
            json.dump(new_tree, f)
    else:
        return new_tree


def jupyter_run_notebook(tree=None, fname=None, outfile=None, rundir='.', 
                         version=4, kernel='python3'):
    """
    Run a Jupyter notebook programatically. The notebook to run can be passed 
    as either a filename or a dict derived from JSON data.
    
    If the notebook experiences an error, a CellExecutionError will be raised. 
    The notebook will still be saved to disk even if it errors though.
    
    Parameters
    ----------
    tree : dict, optional
        Dict containing JSON tree representing a Jupyter notebook.
        
    fname : str, optional
        Filename of Jupyter notebook to load. Only one of 'tree' and 'fname' 
        should be specified.
    
    outfile : str, optional
        File to store Jupyter notebook into after it has run. Default: None 
        (no notebook file will be saved).
    
    rundir : str, optional
        Directory to run the script from. Default: '.' (current directory).
    
    version : int, optional
        Version of Jupyter notebooks to use.
    
    kernel : str, optional
        Name of Jupyter Python kernel to use. Default: 'python3'.
    """
    # Check for valid arguments
    if (tree is None and fname is None) \
    or (tree is not None and fname is not None):
        raise ValueError("Must specify either 'tree' or 'fname'.")
    
    # Load Jupyter notebook as JSON file
    if fname is not None:
        with open(fname, 'r') as f:
            tree = json.load(f)
    
    # Create NotebookNode object needed for execution
    nb = nbformat.reads(json.dumps(tree), as_version=version)
    
    # Validate notebook
    nbformat.validate(nb)
    
    # Initialise notebook preprocessor object
    execp = ExecutePreprocessor(timeout=600, kernel_name=kernel)
    
    # Try to execute notebook; raise error if it fails
    try:
        out = execp.preprocess(nb, {'metadata': {'path': rundir}})
    except CellExecutionError as err:
        raise(err)
    finally:
        # Write notebook file to disk if outfile specified
        if outfile is not None:
            with open(outfile, mode='w') as f:
                nbformat.write(nb, f)
    
