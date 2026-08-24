import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function MarkdownMessage({
  content,
}: {
  content: string;
}) {
  return (
    <div
      className="
        prose prose-invert max-w-none break-words

        prose-p:leading-relaxed
        prose-p:text-gray-300

        prose-headings:break-words
        prose-headings:text-gray-100
        prose-headings:font-semibold

        prose-a:break-all
        prose-a:text-indigo-400
        prose-a:no-underline
        hover:prose-a:underline

        prose-code:break-words
        prose-code:text-cyan-300
        prose-code:bg-cyan-900/20
        prose-code:px-1.5
        prose-code:py-0.5
        prose-code:rounded-md
        prose-code:before:content-none
        prose-code:after:content-none

        prose-pre:max-w-full
        prose-pre:overflow-x-auto
        prose-pre:bg-[#0C0C0F]
        prose-pre:border
        prose-pre:border-white/10
        prose-pre:shadow-2xl
        prose-pre:rounded-xl
        prose-pre:mt-4
        prose-pre:mb-6

        prose-li:text-gray-300

        prose-strong:text-white
        prose-strong:font-semibold

        prose-table:block
        prose-table:max-w-full
        prose-table:overflow-x-auto
      "
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </div>
  );
}