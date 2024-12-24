import os
from document_image_processors import DocumentFormatConverter, DocumentImageResizer
from document_image_pipelines import DocumentImageGptPipeline

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def check_processors():
    """
    Check if the processors are correctly implemented.
    """
    # Initialize processors
    processors = [DocumentFormatConverter(), DocumentImageResizer()]
    
    # Check if the processors are correct
    assert all(isinstance(processor, (DocumentFormatConverter, DocumentImageResizer)) for processor in processors), "Incorrect processors."
    
    # Print message
    print("Processors are correctly implemented.")


if __name__ == "__main__":
    # Initialize pipeline
    check_processors()
    pipeline = DocumentImageGptPipeline()
    
    
    # Print output
    print(pipeline)