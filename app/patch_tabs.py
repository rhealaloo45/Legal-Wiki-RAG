import re

HTML_PATH = "/Users/rhea/Desktop/Rhea Code/Legal-Wiki-RAG/app/templates/index.html"

def patch_html():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    # Hide tabs initially
    html = html.replace('<ul class="nav nav-pills mb-4" id="modeTabs" role="tablist">',
                        '<ul class="nav nav-pills mb-4 d-none" id="modeTabs" role="tablist">')

    # Show tabs on resumeSession
    resume_target = "uploadDone = true;"
    resume_replacement = """uploadDone = true;
    document.getElementById('modeTabs').classList.remove('d-none');"""
    if "document.getElementById('modeTabs').classList.remove('d-none');" not in html.split("async function resumeSession")[1].split("catch")[0]:
        html = html.replace(resume_target, resume_replacement, 1)

    # Show tabs on upload completion
    upload_target = "uploadDone = true;"
    upload_replacement = """uploadDone = true;
          document.getElementById('modeTabs').classList.remove('d-none');"""
    # Second occurrence of uploadDone = true is in the upload handler
    # Let's just replace all occurrences of `uploadDone = true;` that don't already have it
    new_html = []
    for line in html.split("\n"):
        if "uploadDone = true;" in line:
            new_html.append(line)
            new_html.append("    document.getElementById('modeTabs')?.classList.remove('d-none');")
        elif "document.getElementById('modeTabs')?.classList.remove('d-none');" not in line and "document.getElementById('modeTabs').classList.remove('d-none');" not in line:
            new_html.append(line)
    
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(new_html))

if __name__ == "__main__":
    patch_html()
    print("Done")
